"""The fused path must not change the objective.

`loss_from_states` re-implements `SlimDucer.forward`'s loss against
pre-encoded towers so the alignment's encodings are reused. That is a copy,
and a copy can drift, so this pins it to the original. The WSD schedule is
checked here too because getting it wrong is silent: an off-by-one at the
phase boundaries would anneal the run early and still look like a normal
loss curve.
"""
from __future__ import annotations

import unittest

import torch

import slimducer
from slimducer import SlimDucer
from slimducer_train_steps import loss_from_states, wsd_scale


class WsdTest(unittest.TestCase):
    def test_phases(self):
        w, s, d = 1000, 10000, 4000
        self.assertAlmostEqual(wsd_scale(1, w, s, d), 0.001)
        self.assertAlmostEqual(wsd_scale(w, w, s, d), 1.0)
        self.assertAlmostEqual(wsd_scale(w + 1, w, s, d), 1.0)
        self.assertAlmostEqual(wsd_scale(w + s, w, s, d), 1.0)
        # first decay step is below one, last is zero, and it never goes
        # negative if the loop overruns by a step
        self.assertLess(wsd_scale(w + s + 1, w, s, d), 1.0)
        self.assertAlmostEqual(wsd_scale(w + s + d, w, s, d), 0.0)
        self.assertGreaterEqual(wsd_scale(w + s + d + 50, w, s, d), 0.0)

    def test_decay_is_linear(self):
        w, s, d = 1000, 10000, 4000
        mid = wsd_scale(w + s + d // 2, w, s, d)
        self.assertAlmostEqual(mid, 0.5, places=6)


class FusedLossTest(unittest.TestCase):
    """The fused loss equals the original on the same alignment."""

    @classmethod
    def setUpClass(cls):
        cls.dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(0)
        cls.model = SlimDucer(dtype=torch.float32, freeze_audio=True,
                              branch_norm=True).to(cls.dev).eval()

    def test_matches_forward(self):
        m = self.model
        b, t, l = 2, 40, 6
        d_aut = m.joint.from_aut.in_features
        h_aut = [torch.randn(t, d_aut, device=self.dev) for _ in range(b)]
        labels = torch.randint(0, 1000, (b, l), device=self.dev)
        l_lens = torch.tensor([l, l - 2], device=self.dev)
        h_llm = m.encode_text(labels, 1)

        paths = []
        for k in range(b):
            p = torch.full((t,), -1, dtype=torch.long, device=self.dev)
            n = int(l_lens[k])
            p[torch.arange(n, device=self.dev) * 3 + 1] = torch.arange(
                n, device=self.dev)
            paths.append(p)

        fused = loss_from_states(m, h_aut, h_llm, labels, l_lens, paths)

        # the reference, computed the way SlimDucer.forward does it
        import torch.nn.functional as F
        blank_terms, token_terms = [], []
        n_frames = n_tokens = 0
        for k, path in enumerate(paths):
            emitted = (path >= 0).long()
            n_before = torch.cat([emitted.new_zeros(1), emitted.cumsum(0)[:-1]])
            h = m.joint(h_llm[k].index_select(0, n_before), h_aut[k])
            lb = m.blank_head(h).squeeze(-1).float()
            is_blank = path < 0
            blank_terms.append(F.binary_cross_entropy_with_logits(
                lb, is_blank.to(lb.dtype), reduction="sum"))
            ei = (~is_blank).nonzero(as_tuple=True)[0]
            tgt = labels[k].index_select(0, path[ei])
            token_terms.append(F.cross_entropy(
                m.lm_head(h.index_select(0, ei)).float(), tgt, reduction="sum"))
            n_tokens += int(ei.numel())
            n_frames += t
        want = (torch.stack(blank_terms).sum()
                + torch.stack(token_terms).sum()) / n_frames

        self.assertEqual(fused.n_frames, n_frames)
        self.assertEqual(fused.n_tokens, n_tokens)
        self.assertTrue(torch.allclose(fused.loss, want, atol=1e-6),
                        f"{fused.loss.item()} vs {want.item()}")

    def test_gradient_reaches_the_joint(self):
        """A reused encoding still has to carry gradient to the joint."""
        m = self.model
        t, l = 30, 4
        d_aut = m.joint.from_aut.in_features
        h_aut = [torch.randn(t, d_aut, device=self.dev)]
        labels = torch.randint(0, 1000, (1, l), device=self.dev)
        l_lens = torch.tensor([l], device=self.dev)
        with torch.no_grad():
            h_llm = m.encode_text(labels, 1)
        p = torch.full((t,), -1, dtype=torch.long, device=self.dev)
        p[torch.arange(l, device=self.dev) * 3 + 1] = torch.arange(
            l, device=self.dev)

        for q in m.trainable_parameters():
            q.grad = None
        loss_from_states(m, h_aut, h_llm, labels, l_lens, [p]).loss.backward()
        grads = [q.grad for q in m.trainable_parameters() if q.grad is not None]
        self.assertTrue(grads, "no trainable parameter received a gradient")
        self.assertTrue(any(float(g.abs().sum()) > 0 for g in grads))

    def test_lm_head_stays_frozen(self):
        self.assertFalse(any(p.requires_grad
                             for p in self.model.lm_head.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class UniformAlignmentTest(unittest.TestCase):
    """Even spacing still has to be a legal alignment.

    The objective assumes every one of the L tokens is emitted exactly once,
    in order, at a distinct frame. An initialiser that violates that does not
    merely start badly -- it silently drops emission terms from the loss.
    """

    class _Model:
        @staticmethod
        def audio_out_len(mel_len):
            return max(1, int(mel_len * 0.13))

    def _run(self, items):
        import slimducer_train_steps as m
        return m.uniform_alignment(items, self._Model(), None, "cpu",
                                   100000, 100000)[0]

    def test_monotonic_and_complete(self):
        items = [{"dur": d, "ids": list(range(n))}
                 for d, n in [(12.7, 41), (2.0, 3), (20.0, 90), (5.5, 17)]]
        align = self._run(items)
        for i, it in enumerate(items):
            p = align[i]
            emitted = p[p >= 0].tolist()
            self.assertEqual(emitted, list(range(len(it["ids"]))),
                             f"utterance {i}: tokens not placed once in order")
            pos = (p >= 0).nonzero().flatten().tolist()
            self.assertEqual(pos, sorted(set(pos)),
                             f"utterance {i}: frames not strictly increasing")

    def test_spread_is_near_uniform(self):
        items = [{"dur": 12.7, "ids": list(range(41))} for _ in range(20)]
        align = self._run(items)
        p = align[0]
        q = (p >= 0).nonzero().flatten().float()
        self.assertAlmostEqual(float(q.std() / len(p)), 0.289, delta=0.02)

    def test_rejects_more_tokens_than_frames(self):
        """A transcript that cannot fit must come back all blank, not truncated."""
        items = [{"dur": 1.0, "ids": list(range(50))}]
        p = self._run(items)[0]
        self.assertTrue(bool((p < 0).all()))
