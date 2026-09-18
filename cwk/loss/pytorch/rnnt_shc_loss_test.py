"""Tests for `rnnt_shc_loss`.

The tests that matter are the ones that pin the forward-backward down
against something independent, because that is where CTC's two silent
bugs lived (padding dominating the normalization, and the uniform
component leaking onto padded positions -- see this repo's CLAUDE.md).
Three independent references are used:

  * brute force -- enumerate every alignment of a tiny lattice and sum,
    which checks log P AND the transition posteriors,
  * torchaudio.functional.rnnt_loss -- checks log P and the gradient at
    alpha = 0 on realistic shapes,
  * the smoothing functions' own endpoints -- beta = 1 must equal
    textbook uniform LS and beta = 0 must equal active-support.

Run as `python rnnt_shc_loss_test.py` (pytest is not installed in
py3_12_sets, and the rest of this package's suites are unittest anyway).
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import itertools
import unittest

import torch

from cwk.loss.pytorch import rnnt_shc_loss


def _brute_force_log_prob(log_probs, labels, blank=0):
    """log P(y | x) by summing over every alignment of one sample.

    An RNN-T alignment is a lattice path from (0, 0) to (T-1, U) using T
    blank transitions and U label transitions; equivalently, a choice of
    which frames the U labels are emitted at. Enumerating the positions of
    the label emissions is therefore an exact enumeration of the paths.

    Args:
        log_probs: (T, U+1, C) log-softmax output for one sample.
        labels: (U,) label ids.
        blank: blank id.

    Returns:
        Python float log P.
    """
    t_len, u1, _ = log_probs.shape
    u_len = u1 - 1
    total = None
    # A path is determined by the sequence of (t, u) it visits. Walk the
    # lattice recursively; T and U are tiny in the tests.
    def walk(t, u, acc):
        nonlocal total
        if t == t_len - 1 and u == u_len:
            v = acc + float(log_probs[t, u, blank])
            total = v if total is None else torch.logaddexp(
                torch.tensor(total), torch.tensor(v)).item()
            return
        if t < t_len - 1:
            walk(t + 1, u, acc + float(log_probs[t, u, blank]))
        if u < u_len:
            walk(t, u + 1, acc + float(log_probs[t, u, int(labels[u])]))
    walk(0, 0, 0.0)
    return total


def _brute_force_posteriors(log_probs, labels, blank=0):
    """Transition posteriors by enumeration, for cross-checking alpha/beta.

    Returns:
        (q_blank, q_label) each (T, U+1) in probability domain.
    """
    t_len, u1, _ = log_probs.shape
    u_len = u1 - 1
    log_z = _brute_force_log_prob(log_probs, labels, blank)
    qb = torch.zeros(t_len, u1, dtype=torch.float64)
    ql = torch.zeros(t_len, u1, dtype=torch.float64)

    def walk(t, u, acc, taken):
        if t == t_len - 1 and u == u_len:
            w = acc + float(log_probs[t, u, blank])
            p = torch.exp(torch.tensor(w - log_z, dtype=torch.float64))
            for (tt, uu, kind) in taken + [(t, u, "b")]:
                (qb if kind == "b" else ql)[tt, uu] += p
            return
        if t < t_len - 1:
            walk(t + 1, u, acc + float(log_probs[t, u, blank]),
                 taken + [(t, u, "b")])
        if u < u_len:
            walk(t, u + 1, acc + float(log_probs[t, u, int(labels[u])]),
                 taken + [(t, u, "l")])
    walk(0, 0, 0.0, [])
    return qb, ql


def _random_case(b, t_len, u_len, c, seed=0, ragged=False):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(b, t_len, u_len + 1, c, generator=g,
                         dtype=torch.float64)
    labels = torch.randint(1, c, (b, u_len), generator=g)
    if ragged:
        logit_lens = torch.randint(max(1, t_len - 2), t_len + 1, (b,),
                                   generator=g)
        target_lens = torch.randint(max(1, u_len - 1), u_len + 1, (b,),
                                    generator=g)
        # torchaudio's rnnt_loss requires logits' U+1 axis to equal
        # max(target_lengths) + 1 and rejects the batch otherwise
        # ("output length mismatch"), so one sample must be full width.
        logit_lens[0] = t_len
        target_lens[0] = u_len
    else:
        logit_lens = torch.full((b,), t_len, dtype=torch.long)
        target_lens = torch.full((b,), u_len, dtype=torch.long)
    return logits, labels, logit_lens, target_lens


class ForwardBackwardTest(unittest.TestCase):

    def test_log_prob_matches_brute_force(self):
        """log P from the recursions equals an exhaustive path sum."""
        for (t_len, u_len, c) in [(3, 2, 4), (4, 1, 3), (5, 3, 5), (2, 2, 3)]:
            logits, labels, tl, ul = _random_case(
                1, t_len, u_len, c, seed=t_len * 10 + u_len)
            lp = torch.log_softmax(logits[0], dim=-1)
            expected = _brute_force_log_prob(lp, labels[0])
            got = -rnnt_shc_loss.RnntShcLoss.apply(
                labels, ul, logits, tl, 0, 0.0, 0.0, "fixed")
            self.assertAlmostEqual(float(got[0]), expected, places=5,
                                   msg=f"T={t_len} U={u_len} C={c}")

    def test_transition_posteriors_match_brute_force(self):
        """alpha/beta reproduce the enumerated transition posteriors."""
        t_len, u_len, c = 4, 2, 4
        logits, labels, tl, ul = _random_case(1, t_len, u_len, c, seed=7)
        lp = torch.log_softmax(logits[0], dim=-1)
        qb_ref, ql_ref = _brute_force_posteriors(lp, labels[0])

        log_probs = torch.log_softmax(logits, dim=-1)
        log_p_blank = log_probs[..., 0]
        lab = labels.clamp(min=0)
        gathered = torch.gather(
            log_probs[:, :, :u_len, :], 3,
            lab.view(1, 1, u_len, 1).expand(1, t_len, u_len, 1)).squeeze(3)
        log_p_label = torch.cat(
            [gathered, torch.full((1, t_len, 1), rnnt_shc_loss.LOG_0,
                                  dtype=torch.float64)], dim=2)
        la, lb, lz = rnnt_shc_loss.calculate_rnnt_alpha_beta(
            log_p_blank, log_p_label, tl, ul)
        qb, ql = rnnt_shc_loss.rnnt_transition_posteriors(
            la, lb, log_p_blank, log_p_label, lz, tl, ul)

        torch.testing.assert_close(qb[0].double(), qb_ref, atol=1e-5,
                                   rtol=1e-4)
        # The last column of q_label is structurally impossible.
        torch.testing.assert_close(ql[0, :, :u_len].double(),
                                   ql_ref[:, :u_len], atol=1e-5, rtol=1e-4)

    def test_occupancy_sums_to_path_length(self):
        """Total transition mass equals T + U, one per step of any path."""
        logits, labels, tl, ul = _random_case(3, 6, 3, 5, seed=11)
        log_probs = torch.log_softmax(logits, dim=-1)
        log_p_blank = log_probs[..., 0]
        lab = labels.clamp(min=0)
        gathered = torch.gather(
            log_probs[:, :, :3, :], 3,
            lab.view(3, 1, 3, 1).expand(3, 6, 3, 1)).squeeze(3)
        log_p_label = torch.cat(
            [gathered, torch.full((3, 6, 1), rnnt_shc_loss.LOG_0,
                                  dtype=torch.float64)], dim=2)
        la, lb, lz = rnnt_shc_loss.calculate_rnnt_alpha_beta(
            log_p_blank, log_p_label, tl, ul)
        qb, ql = rnnt_shc_loss.rnnt_transition_posteriors(
            la, lb, log_p_blank, log_p_label, lz, tl, ul)
        total = (qb + ql).sum(dim=(1, 2))
        torch.testing.assert_close(
            total.double(), (tl + ul).double(), atol=1e-4, rtol=1e-4)


class TorchaudioAgreementTest(unittest.TestCase):

    def setUp(self):
        try:
            from torchaudio.functional import rnnt_loss  # noqa: F401
        except Exception as exc:                          # pragma: no cover
            self.skipTest(f"torchaudio rnnt_loss unavailable: {exc}")

    def test_loss_matches_torchaudio(self):
        """alpha = 0 must reproduce the reference RNN-T loss."""
        from torchaudio.functional import rnnt_loss
        logits, labels, tl, ul = _random_case(4, 12, 5, 7, seed=3,
                                              ragged=True)
        logits32 = logits.float()
        ref = rnnt_loss(logits32, labels.int(), tl.int(), ul.int(),
                        blank=0, reduction="none")
        got = rnnt_shc_loss.RnntShcLoss.apply(
            labels, ul, logits32, tl, 0, 0.0, 0.0, "fixed")
        torch.testing.assert_close(got, ref, atol=2e-3, rtol=2e-4)

    def test_gradient_matches_torchaudio(self):
        """alpha = 0 must reproduce the reference RNN-T gradient."""
        from torchaudio.functional import rnnt_loss
        logits, labels, tl, ul = _random_case(3, 10, 4, 6, seed=5,
                                              ragged=True)
        a = logits.float().clone().requires_grad_(True)
        rnnt_loss(a, labels.int(), tl.int(), ul.int(), blank=0,
                  reduction="sum").backward()
        b = logits.float().clone().requires_grad_(True)
        rnnt_shc_loss.RnntShcLoss.apply(
            labels, ul, b, tl, 0, 0.0, 0.0, "fixed").sum().backward()
        torch.testing.assert_close(b.grad, a.grad, atol=2e-4, rtol=2e-3)


class SmoothingTest(unittest.TestCase):

    def _grad(self, logits, labels, tl, ul, **kw):
        x = logits.float().clone().requires_grad_(True)
        rnnt_shc_loss.RnntShcLoss.apply(
            labels, ul, x, tl, 0, kw.get("alpha", 0.0), kw.get("beta", 0.0),
            kw.get("alpha_mode", "fixed")).sum().backward()
        return x.grad

    def test_alpha_zero_is_unsmoothed(self):
        """Every mode collapses to the plain loss at alpha = 0."""
        logits, labels, tl, ul = _random_case(2, 8, 3, 6, seed=13)
        base = self._grad(logits, labels, tl, ul, alpha=0.0)
        for mode in ("fixed", "active_support", "floored_active_support",
                     "frame_label_support", "asap"):
            g = self._grad(logits, labels, tl, ul, alpha=0.0,
                           alpha_mode=mode)
            torch.testing.assert_close(g, base, atol=0, rtol=0,
                                       msg=f"mode={mode}")

    def test_fas_beta_one_is_uniform_label_smoothing(self):
        """beta = 1 must equal textbook uniform LS on the node target."""
        logits, labels, tl, ul = _random_case(2, 8, 3, 6, seed=17)
        alpha = 0.2
        g = self._grad(logits, labels, tl, ul, alpha=alpha, beta=1.0,
                       alpha_mode="floored_active_support")
        # Reference: build the node target by hand and mix with 1/C.
        ref = self._uniform_ls_gradient(logits, labels, tl, ul, alpha)
        torch.testing.assert_close(g, ref, atol=1e-5, rtol=1e-4)

    def _uniform_ls_gradient(self, logits, labels, tl, ul, alpha):
        logits32 = logits.float()
        log_probs = torch.log_softmax(logits32, dim=-1)
        b, t_len, u1, c = logits32.shape
        log_p_blank = log_probs[..., 0]
        lab = labels.clamp(min=0)
        gathered = torch.gather(
            log_probs[:, :, :u1 - 1, :], 3,
            lab.view(b, 1, u1 - 1, 1).expand(b, t_len, u1 - 1, 1)).squeeze(3)
        log_p_label = torch.cat(
            [gathered, torch.full((b, t_len, 1), rnnt_shc_loss.LOG_0)], dim=2)
        la, lb, lz = rnnt_shc_loss.calculate_rnnt_alpha_beta(
            log_p_blank, log_p_label, tl, ul)
        qb, ql = rnnt_shc_loss.rnnt_transition_posteriors(
            la, lb, log_p_blank, log_p_label, lz, tl, ul)
        target, gamma = rnnt_shc_loss._node_target(qb, ql, lab, c, 0)
        smoothed = (1.0 - alpha) * target + alpha / c
        return gamma.unsqueeze(3) * (log_probs.exp() - smoothed)

    def test_fas_beta_zero_equals_active_support(self):
        """beta = 0 must equal the active-support mode bit for bit."""
        logits, labels, tl, ul = _random_case(2, 8, 3, 6, seed=19)
        a = self._grad(logits, labels, tl, ul, alpha=0.25, beta=0.0,
                       alpha_mode="floored_active_support")
        b = self._grad(logits, labels, tl, ul, alpha=0.25,
                       alpha_mode="active_support")
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_smoothed_target_stays_a_distribution(self):
        """The smoothed node target must stay non-negative and sum to 1."""
        logits, labels, tl, ul = _random_case(2, 6, 3, 8, seed=23)
        logits32 = logits.float()
        log_probs = torch.log_softmax(logits32, dim=-1)
        b, t_len, u1, c = logits32.shape
        log_p_blank = log_probs[..., 0]
        lab = labels.clamp(min=0)
        gathered = torch.gather(
            log_probs[:, :, :u1 - 1, :], 3,
            lab.view(b, 1, u1 - 1, 1).expand(b, t_len, u1 - 1, 1)).squeeze(3)
        log_p_label = torch.cat(
            [gathered, torch.full((b, t_len, 1), rnnt_shc_loss.LOG_0)], dim=2)
        la, lb, lz = rnnt_shc_loss.calculate_rnnt_alpha_beta(
            log_p_blank, log_p_label, tl, ul)
        qb, ql = rnnt_shc_loss.rnnt_transition_posteriors(
            la, lb, log_p_blank, log_p_label, lz, tl, ul)
        target, _ = rnnt_shc_loss._node_target(qb, ql, lab, c, 0)
        from cwk.loss.pytorch import shc_loss_util
        flat = target.reshape(b, t_len * u1, c)
        full = torch.full((b,), t_len * u1, dtype=torch.long)
        for alpha, beta in itertools.product([0.1, 0.3, 0.6], [0.0, 0.5, 1.0]):
            out = shc_loss_util.apply_floored_active_support_smoothing(
                flat, full, alpha, beta)
            self.assertGreaterEqual(float(out.min()), -1e-6,
                                    f"alpha={alpha} beta={beta}")
            # Only nodes the lattice can actually occupy carry a target.
            # A padded node's row is all zeros, so its "active set" is
            # empty and the floor alone leaves a row summing to
            # alpha*beta; those nodes are zeroed by gamma at the
            # gradient, not here, so they are excluded rather than
            # asserted on.
            live = flat.sum(-1) > 0.5
            sums = out.sum(-1)[live]
            torch.testing.assert_close(
                sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5,
                msg=f"alpha={alpha} beta={beta}")


class PaddingTest(unittest.TestCase):
    """CTC's two bugs were both padding bugs; check the analogues here."""

    def test_batching_does_not_change_a_sample(self):
        """A sample's loss and gradient must not depend on its batch-mates."""
        t_len, u_len, c = 9, 4, 6
        logits, labels, tl, ul = _random_case(3, t_len, u_len, c, seed=29,
                                              ragged=True)
        logits32 = logits.float()
        x = logits32.clone().requires_grad_(True)
        rnnt_shc_loss.RnntShcLoss.apply(
            labels, ul, x, tl, 0, 0.2, 0.0,
            "floored_active_support").sum().backward()

        for i in range(3):
            ti, ui = int(tl[i]), int(ul[i])
            xi = logits32[i:i + 1, :ti, :ui + 1].clone().requires_grad_(True)
            li = rnnt_shc_loss.RnntShcLoss.apply(
                labels[i:i + 1, :ui], ul[i:i + 1], xi, tl[i:i + 1], 0,
                0.2, 0.0, "floored_active_support")
            li.sum().backward()
            torch.testing.assert_close(
                x.grad[i, :ti, :ui + 1], xi.grad[0], atol=1e-5, rtol=1e-4,
                msg=f"sample {i} gradient changed with batching")

    def test_gradient_is_zero_outside_the_valid_rectangle(self):
        """Padded frames and padded label columns must get no gradient."""
        logits, labels, tl, ul = _random_case(3, 10, 5, 6, seed=31,
                                              ragged=True)
        x = logits.float().clone().requires_grad_(True)
        rnnt_shc_loss.RnntShcLoss.apply(
            labels, ul, x, tl, 0, 0.2, 0.3,
            "floored_active_support").sum().backward()
        for i in range(3):
            ti, ui = int(tl[i]), int(ul[i])
            if ti < logits.shape[1]:
                self.assertEqual(float(x.grad[i, ti:].abs().max()), 0.0,
                                 f"sample {i} padded frames got gradient")
            if ui + 1 < logits.shape[2]:
                self.assertEqual(
                    float(x.grad[i, :, ui + 1:].abs().max()), 0.0,
                    f"sample {i} padded label columns got gradient")

    def test_long_sequence_does_not_collapse_to_all_blank(self):
        """The CTC failure mode: a long, low-confidence sample underflows.

        In CTC, padded label positions floored at a FINITE log-zero won
        the logsumexp once the real path's score dropped below it, making
        the target "blank with probability 1 everywhere". Here the target
        is normalized per node over exactly two live entries, so the same
        sample must still put real mass on the label transition.
        """
        t_len, u_len, c = 400, 30, 32
        g = torch.Generator().manual_seed(37)
        # Near-uniform logits, i.e. an untrained head: every path scores
        # about -log(C) per step, so log P is about -(T+U) log C = -1490.
        logits = (0.01 * torch.randn(1, t_len, u_len + 1, c, generator=g)
                  ).double()
        labels = torch.randint(1, c, (1, u_len), generator=g)
        tl = torch.tensor([t_len])
        ul = torch.tensor([u_len])
        loss = rnnt_shc_loss.RnntShcLoss.apply(
            labels, ul, logits, tl, 0, 0.0, 0.0, "fixed")
        self.assertTrue(torch.isfinite(loss).all(), "loss went non-finite")
        log_probs = torch.log_softmax(logits, dim=-1)
        log_p_blank = log_probs[..., 0]
        gathered = torch.gather(
            log_probs[:, :, :u_len, :], 3,
            labels.view(1, 1, u_len, 1).expand(1, t_len, u_len, 1)).squeeze(3)
        log_p_label = torch.cat(
            [gathered, torch.full((1, t_len, 1), rnnt_shc_loss.NEG_INF,
                                  dtype=torch.float64)], dim=2)
        la, lb, lz = rnnt_shc_loss.calculate_rnnt_alpha_beta(
            log_p_blank, log_p_label, tl, ul)
        qb, ql = rnnt_shc_loss.rnnt_transition_posteriors(
            la, lb, log_p_blank, log_p_label, lz, tl, ul)
        # Every one of the U labels has to be emitted exactly once and
        # every one of the T frames consumes exactly one blank, so these
        # totals are U and T no matter how unconfident the model is.
        # Asserted in float64; the float32 path drifts by ~5e-4 relative
        # over a lattice this long (see calculate_rnnt_alpha_beta).
        self.assertAlmostEqual(float(ql.sum()), float(u_len), delta=1e-6)
        self.assertAlmostEqual(float(qb.sum()), float(t_len), delta=1e-6)
        # float32 must still be finite and close, just not exact.
        la32, lb32, lz32 = rnnt_shc_loss.calculate_rnnt_alpha_beta(
            log_p_blank.float(), log_p_label.float(), tl, ul)
        qb32, ql32 = rnnt_shc_loss.rnnt_transition_posteriors(
            la32, lb32, log_p_blank.float(), log_p_label.float(), lz32,
            tl, ul)
        self.assertTrue(torch.isfinite(qb32).all() and
                        torch.isfinite(ql32).all())
        self.assertAlmostEqual(float(qb32.sum()) / t_len, 1.0, delta=2e-3)
        self.assertAlmostEqual(float(ql32.sum()) / u_len, 1.0, delta=2e-3)

class FrameLabelSupportTest(unittest.TestCase):
    """`frame_label_support`: smoothing confined to the label subspace.

    The mode exists because per-node FAS raises the blank target by
    alpha/K at every node where a label has to be emitted, which in RNN-T
    is that label's only chance to be emitted. The tests below pin the two
    claims that make the mode worth having: the blank target does not
    move, and the active set is genuinely multi-class (which a per-node
    blank-excluded set could never be, since it would hold one label).
    """

    def _target(self, logits, labels, tl, ul, **kw):
        """Returns (plain target, smoothed target, gamma)."""
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        b, t_len, u1, c = logits.shape
        log_p_blank = log_probs[..., 0]
        lab = labels.clamp(min=0)
        gathered = torch.gather(
            log_probs[:, :, :u1 - 1, :], 3,
            lab.view(b, 1, u1 - 1, 1).expand(b, t_len, u1 - 1, 1)).squeeze(3)
        log_p_label = torch.cat(
            [gathered, torch.full((b, t_len, 1), rnnt_shc_loss.NEG_INF)],
            dim=2)
        la, lb, lz = rnnt_shc_loss.calculate_rnnt_alpha_beta(
            log_p_blank, log_p_label, tl, ul)
        qb, ql = rnnt_shc_loss.rnnt_transition_posteriors(
            la, lb, log_p_blank, log_p_label, lz, tl, ul)
        target, gamma = rnnt_shc_loss._node_target(qb, ql, lab, c, 0)
        smoothed = rnnt_shc_loss.apply_frame_label_support_smoothing(
            target, gamma, 0, kw.get("alpha", 0.2), kw.get("beta", 0.0),
            eps=kw.get("eps", 1e-10))
        return target, smoothed, gamma

    def test_blank_target_is_unchanged(self):
        """The defining property: d target(blank) = 0 at every node.

        This is the whole point of the mode -- per-node FAS moves it by
        (alpha/K)(1 - N_a q_b), which is +alpha/K exactly where a label
        must be emitted.
        """
        logits, labels, tl, ul = _random_case(2, 8, 3, 6, seed=31)
        target, smoothed, _ = self._target(logits, labels, tl, ul, alpha=0.3)
        torch.testing.assert_close(smoothed[..., 0], target[..., 0],
                                   atol=1e-6, rtol=1e-5)

    def test_per_node_fas_does_move_the_blank_target(self):
        """Control for the test above: the mode it replaces does move it.

        Without this the previous test could pass for the trivial reason
        that the case has no label mass anywhere.
        """
        logits, labels, tl, ul = _random_case(2, 8, 3, 6, seed=31)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        b, t_len, u1, c = logits.shape
        target, _, gamma = self._target(logits, labels, tl, ul, alpha=0.3)
        flat = target.reshape(b, t_len * u1, c)
        full = torch.full((b,), t_len * u1, dtype=torch.long)
        from cwk.loss.pytorch import shc_loss_util
        node = shc_loss_util.apply_floored_active_support_smoothing(
            flat, full, 0.3, 0.0).reshape(b, t_len, u1, c)
        moved = (node[..., 0] - target[..., 0]).abs()
        self.assertGreater(float(moved[gamma > 1e-6].max()), 1e-3)

    def test_target_stays_a_distribution(self):
        """Non-negative and summing to 1 wherever the node is occupied."""
        for beta in (0.0, 0.3, 1.0):
            logits, labels, tl, ul = _random_case(2, 6, 3, 8, seed=37)
            _, smoothed, gamma = self._target(
                logits, labels, tl, ul, alpha=0.4, beta=beta)
            occupied = gamma > 1e-6
            self.assertGreaterEqual(float(smoothed.min()), -1e-7,
                                    msg=f"beta={beta}")
            sums = smoothed.sum(-1)[occupied]
            torch.testing.assert_close(sums, torch.ones_like(sums),
                                       atol=1e-5, rtol=1e-5)

    def test_active_set_is_frame_level_not_node_level(self):
        """Mass must reach labels other than this node's own y_{u+1}.

        A per-node label support holds exactly one class, so smoothing
        over it would be a pure blank penalty. The frame marginal pulls in
        every label the lattice could emit at this frame, which is what
        makes the mode a smoother at all.
        """
        logits, labels, tl, ul = _random_case(2, 8, 4, 6, seed=41)
        target, smoothed, gamma = self._target(
            logits, labels, tl, ul, alpha=0.3, beta=0.0)
        # Classes that had exactly zero target at a node but gained mass.
        gained = (target < 1e-12) & (smoothed > 1e-9)
        gained = gained & (gamma > 1e-6).unsqueeze(3)
        self.assertGreater(int(gained.sum()), 0)

    def test_beta_one_is_uniform_over_the_non_blank_classes(self):
        """beta = 1 must be textbook LS applied inside the label part."""
        logits, labels, tl, ul = _random_case(2, 8, 3, 6, seed=43)
        alpha, c = 0.25, logits.shape[-1]
        target, smoothed, gamma = self._target(
            logits, labels, tl, ul, alpha=alpha, beta=1.0)
        rate = alpha / c
        mass = rate * (c - 1.0)
        label_part = target.clone()
        label_part[..., 0] = 0.0
        s = label_part.sum(-1, keepdim=True)
        floor = torch.full_like(target, rate)
        floor[..., 0] = 0.0
        ref = (1.0 - mass) * label_part + floor * s
        ref[..., 0] = target[..., 0]
        torch.testing.assert_close(smoothed, ref, atol=1e-6, rtol=1e-5)

    def test_nodes_without_label_mass_are_untouched(self):
        """The u = U column has no label transition, so nothing to smooth."""
        logits, labels, tl, ul = _random_case(2, 8, 3, 6, seed=47)
        target, smoothed, _ = self._target(logits, labels, tl, ul, alpha=0.3)
        for i, u in enumerate(ul.tolist()):
            torch.testing.assert_close(smoothed[i, :, u], target[i, :, u],
                                       atol=1e-6, rtol=1e-5)

    def test_gradient_is_zero_outside_the_valid_rectangle(self):
        logits, labels, tl, ul = _random_case(3, 9, 4, 6, seed=53)
        x = logits.float().clone().requires_grad_(True)
        rnnt_shc_loss.RnntShcLoss.apply(
            labels, ul, x, tl, 0, 0.3, 0.0,
            "frame_label_support").sum().backward()
        for i, (t, u) in enumerate(zip(tl.tolist(), ul.tolist())):
            self.assertEqual(float(x.grad[i, t:].abs().sum()), 0.0)
            self.assertEqual(float(x.grad[i, :, u + 1:].abs().sum()), 0.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
