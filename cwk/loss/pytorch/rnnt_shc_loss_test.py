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
import math
import unittest

import torch

from cwk.loss.pytorch import rnnt_shc_loss
from cwk.loss.pytorch import shc_loss_util


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
                     "frame_label_support", "diagonal_active_support",
                     "alignment_biased", "asap"):
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
class DiagonalActiveSupportTest(unittest.TestCase):
    """`diagonal_active_support`: one active set per anti-diagonal.

    k = t + u is the step index of an alignment path -- every transition
    advances exactly one of the two axes -- so a path visits exactly one
    node per anti-diagonal. That is what makes the tokens crossing a
    diagonal competing hypotheses for the same decision, and it is what a
    per-FRAME set lacks: at fixed t a path climbs several u, so those
    nodes are sequential rather than alternative.
    """

    def _fb(self, logits, labels, tl, ul):
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
        return target, gamma

    def test_occupancy_sums_to_one_on_every_anti_diagonal(self):
        """sum_{t+u=k} gamma(t,u) = 1 for every step k.

        This is the property the whole mode rests on, and the RNN-T
        analogue of CTC's sum_l gamma(t,l) = 1.
        """
        logits, labels, tl, ul = _random_case(3, 9, 4, 6, seed=61)
        _, gamma = self._fb(logits, labels, tl, ul)
        for i, (t, u) in enumerate(zip(tl.tolist(), ul.tolist())):
            g = gamma[i, :t, :u + 1]
            for k in range(t + u):
                idx = [(x, k - x) for x in range(max(0, k - u), min(t, k + 1))]
                tot = sum(float(g[x, y]) for x, y in idx)
                self.assertAlmostEqual(tot, 1.0, delta=1e-5,
                                       msg=f"sample {i}, k={k}")

    def test_diagonal_token_mass_is_a_distribution(self):
        """Each diagonal's token mass sums to 1 over classes."""
        logits, labels, tl, ul = _random_case(3, 9, 4, 6, seed=67)
        target, gamma = self._fb(logits, labels, tl, ul)
        p = rnnt_shc_loss.diagonal_token_mass(target, gamma)
        for i, (t, u) in enumerate(zip(tl.tolist(), ul.tolist())):
            for k in range(t + u):
                self.assertAlmostEqual(float(p[i, k].sum()), 1.0, delta=1e-5,
                                       msg=f"sample {i}, k={k}")

    def test_nodes_on_one_diagonal_get_the_same_floor(self):
        """The defining property: the active set is shared along t + u."""
        logits, labels, tl, ul = _random_case(2, 8, 3, 6, seed=71)
        target, gamma = self._fb(logits, labels, tl, ul)
        flat = torch.zeros_like(target)
        out = rnnt_shc_loss.apply_diagonal_active_support_smoothing(
            flat, gamma, 0.3, 0.0)          # zero target isolates the floor
        for i, (t, u) in enumerate(zip(tl.tolist(), ul.tolist())):
            for k in range(t + u):
                idx = [(x, k - x) for x in range(max(0, k - u), min(t, k + 1))]
                if len(idx) < 2:
                    continue
                first = out[i, idx[0][0], idx[0][1]]
                for x, y in idx[1:]:
                    torch.testing.assert_close(out[i, x, y], first,
                                               atol=0, rtol=0)

    def test_arrival_is_departure_shifted_by_one_diagonal(self):
        """The two alignments differ by exactly one step, nothing else."""
        logits, labels, tl, ul = _random_case(2, 8, 3, 6, seed=73)
        target, gamma = self._fb(logits, labels, tl, ul)
        zero = torch.zeros_like(target)
        dep = rnnt_shc_loss.apply_diagonal_active_support_smoothing(
            zero, gamma, 0.3, 0.0, align="departure")
        arr = rnnt_shc_loss.apply_diagonal_active_support_smoothing(
            zero, gamma, 0.3, 0.0, align="arrival")
        b, t_len, u1, _ = target.shape
        for i in range(b):
            for t in range(t_len):
                for u in range(u1):
                    if t + u == 0:
                        continue
                    src_t, src_u = (t - 1, u) if t >= 1 else (t, u - 1)
                    torch.testing.assert_close(arr[i, t, u],
                                               dep[i, src_t, src_u],
                                               atol=1e-6, rtol=1e-5)

    def test_target_stays_a_distribution(self):
        for beta in (0.0, 0.3, 1.0):
            for align in ("departure", "arrival"):
                logits, labels, tl, ul = _random_case(2, 6, 3, 8, seed=79)
                target, gamma = self._fb(logits, labels, tl, ul)
                out = rnnt_shc_loss.apply_diagonal_active_support_smoothing(
                    target, gamma, 0.4, beta, align=align)
                occupied = gamma > 1e-6
                self.assertGreaterEqual(float(out.min()), -1e-7,
                                        msg=f"beta={beta} {align}")
                sums = out.sum(-1)[occupied]
                torch.testing.assert_close(
                    sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5,
                    msg=f"beta={beta} {align}")

    def test_beta_one_is_textbook_label_smoothing(self):
        """beta = 1 puts alpha/C on every class, so it is plain LS."""
        logits, labels, tl, ul = _random_case(2, 8, 3, 6, seed=83)
        alpha, c = 0.25, logits.shape[-1]
        target, gamma = self._fb(logits, labels, tl, ul)
        out = rnnt_shc_loss.apply_diagonal_active_support_smoothing(
            target, gamma, alpha, 1.0)
        ref = (1.0 - alpha) * target + alpha / c
        occupied = (gamma > 1e-6).unsqueeze(3).expand_as(out)
        torch.testing.assert_close(out[occupied], ref[occupied],
                                   atol=1e-6, rtol=1e-5)

    def test_gradient_is_zero_outside_the_valid_rectangle(self):
        logits, labels, tl, ul = _random_case(3, 9, 4, 6, seed=89)
        x = logits.float().clone().requires_grad_(True)
        rnnt_shc_loss.RnntShcLoss.apply(
            labels, ul, x, tl, 0, 0.3, 0.0,
            "diagonal_active_support").sum().backward()
        for i, (t, u) in enumerate(zip(tl.tolist(), ul.tolist())):
            self.assertEqual(float(x.grad[i, t:].abs().sum()), 0.0)
            self.assertEqual(float(x.grad[i, :, u + 1:].abs().sum()), 0.0)
class AlignmentBiasedTest(unittest.TestCase):
    """`alignment_biased` (ABS): smooth toward the no-acoustics alignment.

    The reference is what the lattice itself says when the model says
    nothing, which for RNN-T is "remaining frames : remaining labels". It
    lives on the same two classes the node's own target does, so it moves
    the blank/label split without ever nominating a label that is wrong
    given the node's predictor history.
    """

    def _uniform_recursion(self, labels, tl, ul, t_len, c):
        """gamma and node target from the recursion with flat acoustics."""
        b = labels.shape[0]
        u1 = labels.shape[1] + 1
        lp = torch.full((b, t_len, u1, c), math.log(1.0 / c),
                        dtype=torch.float64)
        lab = labels.clamp(min=0)
        gathered = torch.gather(
            lp[:, :, :u1 - 1, :], 3,
            lab.view(b, 1, u1 - 1, 1).expand(b, t_len, u1 - 1, 1)).squeeze(3)
        lpl = torch.cat(
            [gathered, torch.full((b, t_len, 1), rnnt_shc_loss.NEG_INF,
                                  dtype=torch.float64)], dim=2)
        la, lb, lz = rnnt_shc_loss.calculate_rnnt_alpha_beta(
            lp[..., 0], lpl, tl, ul)
        qb, ql = rnnt_shc_loss.rnnt_transition_posteriors(
            la, lb, lp[..., 0], lpl, lz, tl, ul)
        return rnnt_shc_loss._node_target(qb, ql, lab, c, 0)

    def test_reference_matches_the_flat_acoustic_recursion(self):
        """The closed form must equal running forward-backward at 1/C.

        This is the whole claim: "remaining frames : remaining labels" IS
        the alignment posterior under uniform acoustics, not an
        approximation of it.
        """
        logits, labels, tl, ul = _random_case(3, 9, 4, 6, seed=101)
        b, t_len, u1, c = logits.shape
        ref_recursion, gamma = self._uniform_recursion(labels, tl, ul, t_len, c)
        ref_closed = rnnt_shc_loss.uniform_acoustic_node_target(
            tl, ul, labels, t_len, u1, c, 0, torch.float64)
        occupied = (gamma > 1e-12).unsqueeze(3).expand_as(ref_closed)
        torch.testing.assert_close(ref_closed[occupied],
                                   ref_recursion[occupied],
                                   atol=1e-10, rtol=1e-8)

    def test_reference_is_a_distribution_on_two_classes(self):
        """Normalized over classes, and supported on {blank, y_{u+1}} only."""
        logits, labels, tl, ul = _random_case(3, 9, 4, 6, seed=103)
        b, t_len, u1, c = logits.shape
        ref = rnnt_shc_loss.uniform_acoustic_node_target(
            tl, ul, labels, t_len, u1, c, 0, torch.float32)
        sums = ref.sum(-1)
        torch.testing.assert_close(sums, torch.ones_like(sums),
                                   atol=1e-6, rtol=1e-5)
        self.assertGreaterEqual(float(ref.min()), 0.0)
        # Nothing outside {blank} u {y_{u+1}}.
        allowed = torch.zeros_like(ref)
        allowed[..., 0] = 1.0
        lab = labels.clamp(min=0)
        u_real = lab.shape[1]
        allowed[:, :, :u_real, :].scatter_(
            3, lab.view(b, 1, u_real, 1).expand(b, t_len, u_real, 1), 1.0)
        self.assertEqual(float((ref * (1.0 - allowed)).abs().sum()), 0.0)

    def test_puts_no_mass_on_a_wrong_history_label(self):
        """The property every earlier wide-support attempt lacked."""
        logits, labels, tl, ul = _random_case(2, 8, 4, 6, seed=107)
        b, t_len, u1, c = logits.shape
        x = logits.float().clone().requires_grad_(True)
        rnnt_shc_loss.RnntShcLoss.apply(
            labels, ul, x, tl, 0, 0.3, 0.0, "alignment_biased").sum().backward()
        base = logits.float().clone().requires_grad_(True)
        rnnt_shc_loss.RnntShcLoss.apply(
            labels, ul, base, tl, 0, 0.0, 0.0, "fixed").sum().backward()
        # The gradient may only move on the two classes the node owns.
        allowed = torch.zeros_like(x.grad)
        allowed[..., 0] = 1.0
        lab = labels.clamp(min=0)
        u_real = lab.shape[1]
        allowed[:, :, :u_real, :].scatter_(
            3, lab.view(b, 1, u_real, 1).expand(b, t_len, u_real, 1), 1.0)
        moved = (x.grad - base.grad) * (1.0 - allowed)
        self.assertLess(float(moved.abs().max()), 1e-6)

    def test_terminal_node_reference_is_all_blank(self):
        """At (T-1, U) both remainders are zero; only the exit blank is legal."""
        logits, labels, tl, ul = _random_case(3, 9, 4, 6, seed=109)
        b, t_len, u1, c = logits.shape
        ref = rnnt_shc_loss.uniform_acoustic_node_target(
            tl, ul, labels, t_len, u1, c, 0, torch.float32)
        for i, (t, u) in enumerate(zip(tl.tolist(), ul.tolist())):
            self.assertAlmostEqual(float(ref[i, t - 1, u, 0]), 1.0, places=6)

    def test_alpha_one_replaces_the_target_entirely(self):
        """alpha = 1 must leave the reference alone as the target."""
        logits, labels, tl, ul = _random_case(2, 8, 3, 6, seed=113)
        b, t_len, u1, c = logits.shape
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        lab = labels.clamp(min=0)
        gathered = torch.gather(
            log_probs[:, :, :u1 - 1, :], 3,
            lab.view(b, 1, u1 - 1, 1).expand(b, t_len, u1 - 1, 1)).squeeze(3)
        lpl = torch.cat(
            [gathered, torch.full((b, t_len, 1), rnnt_shc_loss.NEG_INF)], dim=2)
        la, lb, lz = rnnt_shc_loss.calculate_rnnt_alpha_beta(
            log_probs[..., 0], lpl, tl, ul)
        qb, ql = rnnt_shc_loss.rnnt_transition_posteriors(
            la, lb, log_probs[..., 0], lpl, lz, tl, ul)
        _, gamma = rnnt_shc_loss._node_target(qb, ql, lab, c, 0)
        ref = rnnt_shc_loss.uniform_acoustic_node_target(
            tl, ul, labels, t_len, u1, c, 0, torch.float32)
        x = logits.float().clone().requires_grad_(True)
        rnnt_shc_loss.RnntShcLoss.apply(
            labels, ul, x, tl, 0, 1.0, 0.0, "alignment_biased").sum().backward()
        expected = gamma.unsqueeze(3) * (log_probs.exp() - ref)
        torch.testing.assert_close(x.grad, expected, atol=1e-5, rtol=1e-4)

    def test_gradient_is_zero_outside_the_valid_rectangle(self):
        logits, labels, tl, ul = _random_case(3, 9, 4, 6, seed=127)
        x = logits.float().clone().requires_grad_(True)
        rnnt_shc_loss.RnntShcLoss.apply(
            labels, ul, x, tl, 0, 0.3, 0.0, "alignment_biased").sum().backward()
        for i, (t, u) in enumerate(zip(tl.tolist(), ul.tolist())):
            self.assertEqual(float(x.grad[i, t:].abs().sum()), 0.0)
            self.assertEqual(float(x.grad[i, :, u + 1:].abs().sum()), 0.0)
class ConfidenceGateTest(unittest.TestCase):
    """`gate`: restrict smoothing by the unsmoothed target's confidence."""

    def _grad(self, logits, labels, tl, ul, **kw):
        x = logits.float().clone().requires_grad_(True)
        rnnt_shc_loss.RnntShcLoss.apply(
            labels, ul, x, tl, 0, kw.get("alpha", 0.3), kw.get("beta", 0.0),
            kw.get("alpha_mode", "floored_active_support"), 1e-10, 1e-3,
            "departure", kw.get("gate", "none"),
            kw.get("gate_thresh", 0.9)).sum().backward()
        return x.grad

    def test_gate_none_is_unchanged(self):
        logits, labels, tl, ul = _random_case(2, 8, 3, 6, seed=131)
        a = self._grad(logits, labels, tl, ul, gate="none")
        b = self._grad(logits, labels, tl, ul)
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_threshold_zero_and_one_are_the_endpoints(self):
        """thresh = 0: 'low' smooths nothing, 'high' smooths everything."""
        logits, labels, tl, ul = _random_case(2, 8, 3, 6, seed=137)
        plain = self._grad(logits, labels, tl, ul, alpha=0.0)
        full = self._grad(logits, labels, tl, ul, alpha=0.3)
        lo0 = self._grad(logits, labels, tl, ul, gate="low", gate_thresh=0.0)
        hi0 = self._grad(logits, labels, tl, ul, gate="high", gate_thresh=0.0)
        torch.testing.assert_close(lo0, plain, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(hi0, full, atol=1e-6, rtol=1e-5)

    def test_low_and_high_partition_the_nodes(self):
        """Every node is smoothed by exactly one of the two gates."""
        logits, labels, tl, ul = _random_case(2, 8, 3, 6, seed=139)
        plain = self._grad(logits, labels, tl, ul, alpha=0.0)
        full = self._grad(logits, labels, tl, ul, alpha=0.3)
        lo = self._grad(logits, labels, tl, ul, gate="low", gate_thresh=0.9)
        hi = self._grad(logits, labels, tl, ul, gate="high", gate_thresh=0.9)
        # (lo - plain) + (hi - plain) == full - plain, node by node.
        torch.testing.assert_close(lo + hi - plain, full, atol=1e-5, rtol=1e-4)

    def test_gradient_is_zero_outside_the_valid_rectangle(self):
        logits, labels, tl, ul = _random_case(3, 9, 4, 6, seed=149)
        for gate in ("low", "high"):
            g = self._grad(logits, labels, tl, ul, gate=gate)
            for i, (t, u) in enumerate(zip(tl.tolist(), ul.tolist())):
                self.assertEqual(float(g[i, t:].abs().sum()), 0.0, gate)
                self.assertEqual(float(g[i, :, u + 1:].abs().sum()), 0.0, gate)

if __name__ == "__main__":
    unittest.main(verbosity=2)


class TargetSharpeningTest(unittest.TestCase):
    """`sharpen` is the opposite of smoothing: it moves confident node
    targets toward a one-hot, taking the loss from marginalization over
    alignments toward a hard (Viterbi) alignment."""

    def _logits(self, seed=0, scale=6.0):
        torch.manual_seed(seed)
        return torch.randn(2, 7, 4, 6) * scale

    def _target(self, logits, **kw):
        """Re-derives the node target the forward pass builds, so a test
        can assert on the target rather than only on the loss."""
        captured = {}
        real = rnnt_shc_loss._node_target

        def spy(*a, **k):
            target, gamma = real(*a, **k)
            captured["gamma"] = gamma
            return target, gamma

        rnnt_shc_loss._node_target = spy
        try:
            x = logits.clone().requires_grad_(True)
            loss = rnnt_shc_loss.rnnt_shc_loss(
                self.labels, self.target_lens, x, self.logits_len,
                reduction="sum", **kw)
            loss.backward()
        finally:
            rnnt_shc_loss._node_target = real
        # grad = gamma * (p - target)  =>  target = p - grad / gamma
        gamma = captured["gamma"]
        probs = torch.log_softmax(logits.float(), dim=-1).exp()
        safe = gamma.clamp(min=1e-12).unsqueeze(3)
        return probs - x.grad.float() / safe, gamma

    def setUp(self):
        self.labels = torch.tensor([[1, 2, 3], [2, 3, 1]])
        self.target_lens = torch.tensor([3, 3])
        self.logits_len = torch.tensor([7, 6])

    def test_sharpen_zero_is_a_no_op(self):
        logits = self._logits()
        base = rnnt_shc_loss.rnnt_shc_loss(
            self.labels, self.target_lens, logits, self.logits_len,
            reduction="none")
        same = rnnt_shc_loss.rnnt_shc_loss(
            self.labels, self.target_lens, logits, self.logits_len,
            sharpen=0.0, sharpen_thresh=0.9, reduction="none")
        self.assertTrue(torch.equal(base, same))

    def test_loss_value_is_untouched(self):
        """Sharpening only rewrites the gradient's target; -log P(y|x) is
        still the true marginal."""
        logits = self._logits()
        base = rnnt_shc_loss.rnnt_shc_loss(
            self.labels, self.target_lens, logits, self.logits_len,
            reduction="none")
        sharp = rnnt_shc_loss.rnnt_shc_loss(
            self.labels, self.target_lens, logits, self.logits_len,
            sharpen=1.0, reduction="none")
        self.assertTrue(torch.allclose(base, sharp))

    def test_confident_nodes_become_one_hot(self):
        logits = self._logits()
        plain, gamma = self._target(logits)
        sharp, _ = self._target(logits, sharpen=1.0, sharpen_thresh=0.9)
        live = gamma > 1e-6
        qualifies = (plain.max(dim=3).values > 0.9) & live
        self.assertGreater(int(qualifies.sum()), 0,
                           "test needs some confident nodes")
        got = sharp[qualifies]
        self.assertTrue(torch.allclose(got.max(dim=-1).values,
                                       torch.ones(got.shape[0]), atol=1e-4))
        self.assertTrue(torch.allclose(got.sum(dim=-1),
                                       torch.ones(got.shape[0]), atol=1e-4))

    def test_unconfident_nodes_are_left_alone(self):
        logits = self._logits(scale=0.3)
        plain, gamma = self._target(logits)
        sharp, _ = self._target(logits, sharpen=1.0, sharpen_thresh=0.9)
        live = gamma > 1e-6
        below = (plain.max(dim=3).values <= 0.9) & live
        self.assertGreater(int(below.sum()), 0)
        self.assertTrue(torch.allclose(plain[below], sharp[below], atol=1e-4))

    def test_sharpen_moves_the_argmax_class_up(self):
        """Partial sharpening interpolates, so the dominant class' target
        rises and the other active class' falls."""
        logits = self._logits()
        plain, gamma = self._target(logits)
        half, _ = self._target(logits, sharpen=0.5, sharpen_thresh=0.9)
        live = (gamma > 1e-6) & (plain.max(dim=3).values > 0.9)
        self.assertGreater(int(live.sum()), 0)
        mx_plain = plain.max(dim=3).values[live]
        mx_half = half.max(dim=3).values[live]
        self.assertTrue(torch.all(mx_half > mx_plain - 1e-6))
        self.assertTrue(torch.all(mx_half < 1.0 + 1e-6))

    def test_applies_to_blank_and_label_dominant_nodes_alike(self):
        logits = self._logits()
        plain, gamma = self._target(logits)
        sharp, _ = self._target(logits, sharpen=1.0, sharpen_thresh=0.9)
        live = (gamma > 1e-6) & (plain.max(dim=3).values > 0.9)
        argmax = plain.argmax(dim=3)
        blank_side = live & (argmax == 0)
        label_side = live & (argmax != 0)
        self.assertGreater(int(blank_side.sum()), 0)
        self.assertGreater(int(label_side.sum()), 0)
        for side in (blank_side, label_side):
            got = sharp[side]
            self.assertTrue(torch.allclose(
                got.max(dim=-1).values, torch.ones(got.shape[0]), atol=1e-4))

    def test_backward_arity(self):
        logits = self._logits().requires_grad_(True)
        loss = rnnt_shc_loss.rnnt_shc_loss(
            self.labels, self.target_lens, logits, self.logits_len,
            alpha=0.1, alpha_mode="floored_active_support", gate="low",
            sharpen=1.0, reduction="sum")
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_composes_with_smoothing(self):
        """alpha and sharpen are independent knobs; running both must not
        raise and must leave the loss value alone."""
        logits = self._logits()
        base = rnnt_shc_loss.rnnt_shc_loss(
            self.labels, self.target_lens, logits, self.logits_len,
            reduction="none")
        both = rnnt_shc_loss.rnnt_shc_loss(
            self.labels, self.target_lens, logits, self.logits_len,
            alpha=0.05, alpha_mode="floored_active_support", sharpen=1.0,
            reduction="none")
        self.assertTrue(torch.allclose(base, both))


class DiagonalOccupancyTest(unittest.TestCase):
    """`diagonal_occupancy` is the one mode that leaves the per-node target
    alone and reweights the nodes instead."""

    def setUp(self):
        self.labels = torch.tensor([[1, 2, 3], [2, 3, 1]])
        self.target_lens = torch.tensor([3, 2])
        self.logits_len = torch.tensor([7, 5])
        torch.manual_seed(0)
        self.logits = torch.randn(2, 7, 4, 6) * 4

    def _pieces(self, **kw):
        """Returns (target, gamma) as the forward pass uses them, with the
        smoothing applied to gamma."""
        captured = {}
        real = rnnt_shc_loss.apply_alignment_weight_smoothing

        def spy(gamma, *a, **k):
            out = real(gamma, *a, **k)
            captured["before"] = gamma.clone()
            captured["after"] = out.clone()
            return out

        rnnt_shc_loss.apply_alignment_weight_smoothing = spy
        try:
            x = self.logits.clone().requires_grad_(True)
            loss = rnnt_shc_loss.rnnt_shc_loss(
                self.labels, self.target_lens, x, self.logits_len,
                alpha_mode="aws", reduction="sum",
                fas_eps=kw.pop("fas_eps", 1e-3), **kw)
            loss.backward()
        finally:
            rnnt_shc_loss.apply_alignment_weight_smoothing = real
        return captured, x.grad

    def _diag_sums(self, gamma):
        b, t_len, u1 = gamma.shape
        d = (torch.arange(t_len).view(-1, 1)
             + torch.arange(u1).view(1, -1)).reshape(-1)
        out = torch.zeros(b, t_len + u1)
        return out.index_add_(1, d, gamma.reshape(b, -1))

    def test_alpha_zero_is_a_no_op(self):
        base = rnnt_shc_loss.rnnt_shc_loss(
            self.labels, self.target_lens, self.logits, self.logits_len,
            reduction="none")
        same = rnnt_shc_loss.rnnt_shc_loss(
            self.labels, self.target_lens, self.logits, self.logits_len,
            alpha=0.0, alpha_mode="aws", reduction="none")
        self.assertTrue(torch.equal(base, same))

    def test_loss_value_is_untouched(self):
        base = rnnt_shc_loss.rnnt_shc_loss(
            self.labels, self.target_lens, self.logits, self.logits_len,
            reduction="none")
        sm = rnnt_shc_loss.rnnt_shc_loss(
            self.labels, self.target_lens, self.logits, self.logits_len,
            alpha=0.1, alpha_mode="aws", reduction="none")
        self.assertTrue(torch.allclose(base, sm))

    def test_diagonal_sums_stay_one(self):
        cap, _ = self._pieces(alpha=0.15)
        for name in ("before", "after"):
            sums = self._diag_sums(cap[name])
            live = sums > 1e-6
            self.assertTrue(
                torch.allclose(sums[live], torch.ones(int(live.sum())),
                               atol=1e-4),
                f"{name}: {sums}")

    def test_nodes_outside_the_rectangle_stay_zero(self):
        cap, _ = self._pieces(alpha=0.15, beta=1.0)
        after = cap["after"]
        t = torch.arange(7).view(1, -1, 1)
        u = torch.arange(4).view(1, 1, -1)
        outside = ~((t < self.logits_len.view(-1, 1, 1))
                    & (u <= self.target_lens.view(-1, 1, 1)))
        self.assertGreater(int(outside.sum()), 0)
        self.assertTrue(torch.all(after[outside] == 0.0))

    def test_it_flattens_the_dominant_node(self):
        """Only where the diagonal has a rival. A lone active node absorbs
        the whole mixing mass -- uniform over a one-element set is that
        element -- so it goes UP, which is the same behavior FAS has in
        class space and not a bug."""
        cap, _ = self._pieces(alpha=0.15, beta=0.0)
        before, after = cap["before"], cap["after"]
        b, t_len, u1 = before.shape
        d = (torch.arange(t_len).view(-1, 1)
             + torch.arange(u1).view(1, -1))
        n_act = torch.zeros(b, t_len + u1).index_add_(
            1, d.reshape(-1), (before > 1e-3).reshape(b, -1).float())
        contested = n_act.index_select(1, d.reshape(-1)).reshape(
            b, t_len, u1) >= 2
        big = (before > 0.5) & contested
        self.assertGreater(int(big.sum()), 0)
        self.assertTrue(torch.all(after[big] < before[big]))
        small = (before > 1e-3) & (before < 0.1) & contested
        self.assertGreater(int(small.sum()), 0)
        self.assertTrue(torch.all(after[small] > before[small]))

    def test_the_target_is_not_touched(self):
        """The whole point: z_hat must be bit-identical to the unsmoothed
        run, so no node is ever trained toward a wrong label."""
        def node_target(alpha, mode):
            grabbed = {}
            real = rnnt_shc_loss._node_target

            def spy(*a, **k):
                target, gamma = real(*a, **k)
                grabbed["t"] = target.clone()
                return target, gamma

            rnnt_shc_loss._node_target = spy
            try:
                rnnt_shc_loss.rnnt_shc_loss(
                    self.labels, self.target_lens, self.logits,
                    self.logits_len, alpha=alpha, alpha_mode=mode,
                    reduction="sum")
            finally:
                rnnt_shc_loss._node_target = real
            return grabbed["t"]

        plain = node_target(0.0, "fixed")
        occ = node_target(0.15, "diagonal_occupancy")
        self.assertTrue(torch.equal(plain, occ))

    def test_the_gradient_does_change(self):
        """Leaving z_hat alone is not the same as doing nothing."""
        x0 = self.logits.clone().requires_grad_(True)
        rnnt_shc_loss.rnnt_shc_loss(
            self.labels, self.target_lens, x0, self.logits_len,
            reduction="sum").backward()
        x1 = self.logits.clone().requires_grad_(True)
        rnnt_shc_loss.rnnt_shc_loss(
            self.labels, self.target_lens, x1, self.logits_len,
            alpha=0.15, alpha_mode="aws",
            reduction="sum").backward()
        rel = (x0.grad - x1.grad).norm() / x0.grad.norm()
        self.assertGreater(float(rel), 1e-3)

    def test_beta_one_lifts_the_inactive_nodes(self):
        """Every node inside the rectangle is reachable, so gamma there is
        never exactly zero; what beta controls is whether the nodes BELOW
        eps get any of the mixing mass."""
        b0, _ = self._pieces(alpha=0.15, beta=0.0)
        b1, _ = self._pieces(alpha=0.15, beta=1.0)
        before = b0["before"]
        t = torch.arange(7).view(1, -1, 1)
        u = torch.arange(4).view(1, 1, -1)
        valid = ((t < self.logits_len.view(-1, 1, 1))
                 & (u <= self.target_lens.view(-1, 1, 1)))
        inactive = valid & (before <= 1e-3)
        self.assertGreater(int(inactive.sum()), 0)
        self.assertTrue(torch.all(b1["after"][inactive]
                                  > b0["after"][inactive]))

    def test_beta_zero_leaves_the_inactive_nodes_exactly_alone(self):
        """The mass handed to the active nodes is taken from the active
        nodes, so a node below eps keeps the value the lattice gave it."""
        cap, _ = self._pieces(alpha=0.15, beta=0.0)
        before, after = cap["before"], cap["after"]
        inactive = (before > 0) & (before <= 1e-3)
        self.assertGreater(int(inactive.sum()), 0)
        self.assertTrue(torch.equal(after[inactive], before[inactive]))

    def test_backward_arity(self):
        x = self.logits.clone().requires_grad_(True)
        rnnt_shc_loss.rnnt_shc_loss(
            self.labels, self.target_lens, x, self.logits_len, alpha=0.1,
            alpha_mode="aws", reduction="sum").backward()
        self.assertTrue(torch.isfinite(x.grad).all())


class RnntModelOutputSmoothingTest(unittest.TestCase):
    """MOS on the joint network's output; alpha/beta/gamma/z_hat all follow."""

    def setUp(self):
        self.labels = torch.tensor([[1, 2, 3], [2, 3, 1]])
        self.target_lens = torch.tensor([3, 2])
        self.logits_len = torch.tensor([7, 5])
        torch.manual_seed(0)
        self.logits = torch.randn(2, 7, 4, 6) * 3

    def _run(self, alpha, mode, eps=1e-2):
        x = self.logits.clone().requires_grad_(True)
        loss = rnnt_shc_loss.rnnt_shc_loss(
            self.labels, self.target_lens, x, self.logits_len, alpha=alpha,
            alpha_mode=mode, fas_eps=eps, reduction="sum")
        loss.backward()
        return float(loss), x.grad.clone()

    def test_alpha_zero_is_a_no_op(self):
        a, ga = self._run(0.0, "fixed")
        b, gb = self._run(0.0, "mos")
        self.assertAlmostEqual(a, b, places=5)
        self.assertTrue(torch.allclose(ga, gb, atol=1e-6))

    def test_smoothing_lowers_the_loss(self):
        base, _ = self._run(0.0, "fixed")
        for a in (0.05, 0.1, 0.2):
            mos, _ = self._run(a, "mos")
            self.assertLess(mos, base)

    def test_gradient_is_finite_and_changes(self):
        _, g0 = self._run(0.0, "fixed")
        _, g1 = self._run(0.05, "mos")
        self.assertTrue(torch.isfinite(g1).all())
        self.assertGreater(float((g0 - g1).norm() / g0.norm()), 1e-3)

    def test_nodes_outside_the_rectangle_get_no_gradient(self):
        _, g = self._run(0.05, "mos")
        self.assertTrue(torch.allclose(g[1, 5:], torch.zeros(2, 4, 6),
                                       atol=1e-7))


class DiagonalProjectedTest(unittest.TestCase):
    """`diagonal_projected`: FAS in the anti-diagonal's class space.

    A node's target has at most two non-zero entries, so per-node FAS has
    an active set of 2 and nothing to redistribute. Aggregating along the
    anti-diagonal t + u = k with gamma as the weight gives a genuine class
    distribution -- the distribution of the class emitted at the path's
    k-th step -- FAS runs there, and the result is projected back onto the
    nodes through the label component alone.
    """

    def _fb(self, logits, labels, tl, ul):
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
        return target, gamma, lab

    def _smooth(self, logits, labels, tl, ul, alpha, beta=0.0, eps=1e-10):
        target, gamma, lab = self._fb(logits, labels, tl, ul)
        out = rnnt_shc_loss.apply_diagonal_projected_smoothing(
            target, gamma, lab, tl, ul, 0, alpha, beta, eps=eps)
        return target, gamma, lab, out

    def _diag_aggregate(self, gamma, target, tl, ul):
        """z_k(j) = sum_{t+u=k} gamma(t,u) target(t,u,j), per sample."""
        b, t_len, u1, c = target.shape
        out = []
        for i, (t, u) in enumerate(zip(tl.tolist(), ul.tolist())):
            z = {}
            for tt in range(t):
                for uu in range(u + 1):
                    z.setdefault(tt + uu, torch.zeros(c, dtype=target.dtype))
                    z[tt + uu] = z[tt + uu] + gamma[i, tt, uu] * target[i, tt, uu]
            out.append(z)
        return out

    def test_eps_must_be_positive(self):
        """eps <= 0 activates classes with no node to project back onto."""
        logits, labels, tl, ul = _random_case(2, 8, 3, 6, seed=101)
        target, gamma, lab = self._fb(logits, labels, tl, ul)
        for bad in (0.0, -1e-10, -10.0):
            with self.assertRaises(AssertionError):
                rnnt_shc_loss.apply_diagonal_projected_smoothing(
                    target, gamma, lab, tl, ul, 0, 0.05, 0.0, eps=bad)

    def test_alpha_zero_is_the_identity(self):
        """At alpha = 0 the projection returns what it was given.

        It is an identity only where a diagonal carries each class at a
        single label position, which is the real case: with C = 32 and a
        normal transcript the active nodes on a diagonal hold distinct
        classes. Not bit-identical because blank is rebuilt as
        1 - (label) rather than copied. `forward` skips the call entirely
        at alpha = 0 -- see the end-to-end test, which IS bit-identical.
        """
        logits, labels, tl, ul = _random_case(2, 12, 6, 32, seed=211)
        target, _, _, out = self._smooth(logits, labels, tl, ul, 0.0)
        self.assertTrue(torch.allclose(out, target, atol=1e-7),
                        msg=f"max |d| = {float((out - target).abs().max())}")

    def test_a_repeated_class_on_one_diagonal_is_left_alone(self):
        """alpha = 0 is an identity even where a class repeats.

        This is what splitting the INCREMENT buys. Sharing out a class's
        whole mass in proportion to gamma gives z_tilde_k(j) / G_k(j),
        which is the gamma-weighted MEAN of z_hat over the nodes emitting
        j -- so at alpha = 0 it replaces each node's value with that mean
        instead of leaving it alone, and the distortion neither scales
        with alpha nor vanishes at zero (measured 0.0012 in gamma-weighted
        L1 on a trained 1hr model). C = 6 with U = 3 forces repeats onto
        most diagonals, so that form failed this test by a wide margin.
        """
        logits, labels, tl, ul = _random_case(2, 8, 3, 6, seed=103)
        target, _, _, out = self._smooth(logits, labels, tl, ul, 0.0)
        self.assertTrue(torch.allclose(out, target, atol=1e-7),
                        msg=f"max |d| = {float((out - target).abs().max())}")

    def test_every_node_gets_the_same_label_multiplier(self):
        """The share rule: one coefficient per (diagonal, class),

            p_hat_l = min(alpha_hat_j / alpha_j * p_l, 1),

        applied to the node's real-token probability. So p_hat_l / p_l is
        the SAME for every node on the diagonal emitting j, except where
        the cap at one bites. Blank is not scaled -- it is taken as
        1 - p_hat_l, which is what makes the cap necessary.
        """
        logits, labels, tl, ul = _random_case(1, 8, 3, 6, seed=103)
        labels[0] = torch.tensor([2, 4, 2])      # class 2 at l = 0 and 2
        target, gamma, lab, out = self._smooth(logits, labels, tl, ul, 0.05)
        t_len, u1 = gamma.shape[1], gamma.shape[2]
        seen = 0
        for k in range(t_len + u1 - 1):
            rows = [(t, k - t) for t in range(t_len)
                    if 0 <= k - t < 3 and int(lab[0, k - t]) == 2]
            if len(rows) < 2:
                continue
            vals = [(float(target[0, t, u, 2]), float(out[0, t, u, 2]))
                    for (t, u) in rows]
            if any(o >= 1.0 - 1e-9 or p <= 1e-12 for p, o in vals):
                continue                       # capped, or nothing to scale
            mult = [o / p for p, o in vals]
            seen += 1
            self.assertAlmostEqual(mult[0], mult[1], places=5,
                                   msg=f"diagonal {k}: {mult}")
        self.assertGreater(seen, 0)

    def test_nothing_is_clipped(self):
        """Scaling both entries and renormalizing cannot leave the
        simplex, so no mass is ever thrown away -- unlike a rule that
        writes the label component directly and clips it at one, which on
        a trained 1hr model lost 31 % of the intended class-space change
        at every alpha from 0.005 to 0.10.
        """
        logits, labels, tl, ul = _random_case(3, 9, 4, 8, seed=149)
        _, _, _, out = self._smooth(logits, labels, tl, ul, 0.20)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-9)

    def test_diagonal_class_distribution_sums_to_one(self):
        """sum_j z_k(j) = 1, for free, from sum_{t+u=k} gamma = 1."""
        logits, labels, tl, ul = _random_case(3, 9, 4, 6, seed=107)
        target, gamma, _ = self._fb(logits, labels, tl, ul)
        for z in self._diag_aggregate(gamma, target, tl, ul):
            for k, v in z.items():
                self.assertAlmostEqual(float(v.sum()), 1.0, places=5,
                                       msg=f"diagonal {k}")

    def test_every_node_stays_normalized(self):
        logits, labels, tl, ul = _random_case(3, 9, 4, 6, seed=109)
        _, _, _, out = self._smooth(logits, labels, tl, ul, 0.10)
        for i, (t, u) in enumerate(zip(tl.tolist(), ul.tolist())):
            s = out[i, :t, :u + 1].sum(dim=-1)
            self.assertTrue(torch.allclose(s, torch.ones_like(s), atol=1e-6),
                            msg=f"sample {i}: {s}")

    def test_target_stays_a_probability(self):
        logits, labels, tl, ul = _random_case(3, 9, 4, 6, seed=113)
        _, _, _, out = self._smooth(logits, labels, tl, ul, 0.20)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-9)

    def test_reaggregation_moves_toward_the_smoothed_distribution(self):
        """Renormalizing costs exactness: sum gamma * z_hat' no longer
        lands on z_tilde_k. It moves the right way, though -- every class
        FAS raised comes back higher and every class it lowered comes back
        lower. Measured on a trained 1hr model, the aggregate covers about
        half the intended change (0.50 at alpha = 0.01 and 0.05), against
        0.69 for a rule that writes the component directly and clips. The
        difference is that here the remainder stays inside the node
        instead of being discarded.
        """
        logits, labels, tl, ul = _random_case(3, 9, 4, 6, seed=127)
        target, gamma, lab, out = self._smooth(logits, labels, tl, ul, 0.05)
        b, t_len, u1, c = target.shape
        before = self._diag_aggregate(gamma, target, tl, ul)
        after = self._diag_aggregate(gamma, out, tl, ul)
        agree = 0
        total = 0
        for i in range(b):
            for k, z in before[i].items():
                expected = shc_loss_util.apply_floored_active_support_smoothing(
                    z.view(1, 1, c), torch.ones(1, dtype=torch.long),
                    0.05, 0.0, eps=1e-10).view(c)
                for j in range(c):
                    want = float(expected[j] - z[j])
                    got = float(after[i][k][j] - z[j])
                    if abs(want) < 1e-6:
                        continue
                    total += 1
                    if want * got > 0:
                        agree += 1
        self.assertGreater(total, 50)
        # Measured 0.85 here and 0.86 at C = 32; the minority that move
        # the wrong way are nodes whose renormalizer is dominated by the
        # other entry's coefficient.
        self.assertGreater(agree / total, 0.8,
                           msg=f"{agree}/{total} moved the right way")

    def test_blank_is_never_scaled_directly(self):
        """Blank's new value is 1 - (label), not blank * r_k(blank).

        Scaling blank by its own ratio as well would double-count: the
        node would no longer sum to one and would need a renormalization
        that undoes part of the smoothing.
        """
        logits, labels, tl, ul = _random_case(2, 8, 3, 6, seed=131)
        target, _, lab, out = self._smooth(logits, labels, tl, ul, 0.10)
        b, t_len, u1, c = target.shape
        for i, (t, u) in enumerate(zip(tl.tolist(), ul.tolist())):
            for tt in range(t):
                for uu in range(u):
                    j = int(lab[i, uu])
                    self.assertAlmostEqual(
                        float(out[i, tt, uu, 0]),
                        1.0 - float(out[i, tt, uu, j]), places=6)

    def test_last_column_keeps_an_all_blank_target(self):
        """u = U_b has no label left, so nothing is projected onto it."""
        logits, labels, tl, ul = _random_case(3, 9, 4, 6, seed=137)
        _, _, _, out = self._smooth(logits, labels, tl, ul, 0.15)
        for i, (t, u) in enumerate(zip(tl.tolist(), ul.tolist())):
            col = out[i, :t, u]
            self.assertTrue(torch.allclose(col[:, 0], torch.ones(t,
                                                                dtype=col.dtype),
                                           atol=1e-9), msg=f"{col}")

    def test_outside_the_rectangle_is_untouched(self):
        logits, labels, tl, ul = _random_case(3, 10, 5, 6, seed=139,
                                              ragged=True)
        target, _, _, out = self._smooth(logits, labels, tl, ul, 0.10)
        for i, (t, u) in enumerate(zip(tl.tolist(), ul.tolist())):
            self.assertTrue(torch.equal(out[i, t:], target[i, t:]))
            self.assertTrue(torch.equal(out[i, :, u + 1:],
                                        target[i, :, u + 1:]))

    def test_a_larger_eps_shrinks_the_active_set(self):
        """eps weakens the smoothing by narrowing the active classes."""
        logits, labels, tl, ul = _random_case(3, 9, 4, 8, seed=149)
        target, gamma, _ = self._fb(logits, labels, tl, ul)
        counts = []
        for eps in (1e-10, 1e-3, 1e-1):
            n = 0
            for z in self._diag_aggregate(gamma, target, tl, ul):
                for v in z.values():
                    n += int((v > eps).sum())
            counts.append(n)
        self.assertGreater(counts[0], counts[1])
        self.assertGreater(counts[1], counts[2])

    def test_it_raises_the_diagonal_class_entropy(self):
        """The point of the mode: a flatter per-step class distribution."""
        logits, labels, tl, ul = _random_case(3, 9, 4, 8, seed=151)
        target, gamma, _, out = self._smooth(logits, labels, tl, ul, 0.20)

        def ent(tgt):
            tot = 0.0
            for z in self._diag_aggregate(gamma, tgt, tl, ul):
                for v in z.values():
                    p = v.clamp(min=1e-30)
                    tot += float(-(p * p.log()).sum())
            return tot

        self.assertGreater(ent(out), ent(target))

    def test_end_to_end_gradient_is_finite_and_differs(self):
        logits, labels, tl, ul = _random_case(2, 8, 3, 6, seed=157)
        grads = []
        for alpha in (0.0, 0.10):
            x = logits.clone().requires_grad_(True)
            loss = rnnt_shc_loss.rnnt_shc_loss(
                labels, ul, x, tl, blank=0, alpha=alpha,
                alpha_mode="diagonal_projected", fas_eps=1e-10)
            loss.sum().backward()
            grads.append(x.grad.clone())
        self.assertTrue(torch.isfinite(grads[1]).all())
        self.assertGreater(
            float((grads[0] - grads[1]).norm() / grads[0].norm()), 1e-3)

    def test_end_to_end_alpha_zero_is_bit_identical(self):
        logits, labels, tl, ul = _random_case(2, 8, 3, 6, seed=163)
        outs = []
        for mode in ("fixed", "diagonal_projected"):
            x = logits.clone().requires_grad_(True)
            loss = rnnt_shc_loss.rnnt_shc_loss(
                labels, ul, x, tl, blank=0, alpha=0.0, alpha_mode=mode,
                fas_eps=1e-10)
            loss.sum().backward()
            outs.append((loss.detach().clone(), x.grad.clone()))
        self.assertTrue(torch.equal(outs[0][0], outs[1][0]))
        self.assertTrue(torch.equal(outs[0][1], outs[1][1]))

    def test_nodes_outside_the_rectangle_get_no_gradient(self):
        logits, labels, tl, ul = _random_case(3, 10, 5, 6, seed=167,
                                              ragged=True)
        x = logits.clone().requires_grad_(True)
        loss = rnnt_shc_loss.rnnt_shc_loss(
            labels, ul, x, tl, blank=0, alpha=0.10,
            alpha_mode="diagonal_projected", fas_eps=1e-10)
        loss.sum().backward()
        for i, (t, u) in enumerate(zip(tl.tolist(), ul.tolist())):
            self.assertTrue(torch.allclose(
                x.grad[i, t:], torch.zeros_like(x.grad[i, t:]), atol=1e-9))
            self.assertTrue(torch.allclose(
                x.grad[i, :, u + 1:],
                torch.zeros_like(x.grad[i, :, u + 1:]), atol=1e-9))


class AwsAlphaBetaTest(unittest.TestCase):
    """`aws_alpha_beta`: AWS on both lattice halves, not on gamma.

    Plain AWS smooths gamma = alpha*beta/P, which is a pure per-node
    weight, so the target is untouched by construction. Smoothing the
    halves separately is a different intervention: alpha cancels out of
    z_hat(t,u,blank) = p(blank|t,u) beta(t+1,u) / beta(t,u), so it moves
    only the weight, while beta moves the weight and the target together.
    """

    def _halves(self, logits, labels, tl, ul):
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        b, t_len, u1, c = logits.shape
        lab = labels.clamp(min=0)
        gathered = torch.gather(
            log_probs[:, :, :u1 - 1, :], 3,
            lab.view(b, 1, u1 - 1, 1).expand(b, t_len, u1 - 1, 1)).squeeze(3)
        log_p_label = torch.cat(
            [gathered, torch.full((b, t_len, 1), rnnt_shc_loss.NEG_INF)],
            dim=2)
        return rnnt_shc_loss.calculate_rnnt_alpha_beta(
            log_probs[..., 0], log_p_label, tl, ul)

    def test_alpha_zero_leaves_the_lattice_alone(self):
        logits, labels, tl, ul = _random_case(3, 9, 4, 6, seed=401)
        la, lb, _ = self._halves(logits, labels, tl, ul)
        for x in (la, lb):
            out = rnnt_shc_loss.apply_alpha_beta_diagonal_smoothing(
                x, tl, ul, 0.0, 0.0, eps=1e-10)
            self.assertTrue(torch.allclose(out, x, atol=1e-4),
                            msg=f"max |d| = {float((out - x).abs().max())}")

    def test_it_is_invariant_to_a_per_diagonal_rescale(self):
        """alpha and beta are only defined up to a per-diagonal factor
        that cancels in gamma, so the smoothing must not depend on it."""
        logits, labels, tl, ul = _random_case(2, 8, 3, 6, seed=403)
        la, _, _ = self._halves(logits, labels, tl, ul)
        b, t_len, u1 = la.shape
        t = torch.arange(t_len).view(-1, 1)
        u = torch.arange(u1).view(1, -1)
        g = torch.Generator().manual_seed(9)
        shift = torch.randn(t_len + u1, generator=g, dtype=la.dtype) * 3.0
        bumped = la + shift[(t + u)].unsqueeze(0)
        a = rnnt_shc_loss.apply_alpha_beta_diagonal_smoothing(
            la, tl, ul, 0.05, 0.0, eps=1e-10)
        c = rnnt_shc_loss.apply_alpha_beta_diagonal_smoothing(
            bumped, tl, ul, 0.05, 0.0, eps=1e-10)
        valid = ((t.view(1, -1, 1) < tl.view(-1, 1, 1))
                 & (u.view(1, 1, -1) <= ul.view(-1, 1, 1)))
        d = ((c - shift[(t + u)].unsqueeze(0)) - a)[valid].abs().max()
        self.assertLess(float(d), 1e-3, msg=f"{float(d)}")

    def test_outside_the_rectangle_is_untouched(self):
        logits, labels, tl, ul = _random_case(3, 10, 5, 6, seed=405,
                                              ragged=True)
        la, lb, _ = self._halves(logits, labels, tl, ul)
        for x in (la, lb):
            out = rnnt_shc_loss.apply_alpha_beta_diagonal_smoothing(
                x, tl, ul, 0.05, 0.0, eps=1e-10)
            for i, (t, u) in enumerate(zip(tl.tolist(), ul.tolist())):
                self.assertTrue(torch.equal(out[i, t:], x[i, t:]))
                self.assertTrue(torch.equal(out[i, :, u + 1:],
                                            x[i, :, u + 1:]))

    def test_smoothing_alpha_moves_the_weight_but_not_the_target(self):
        """The asymmetry the mode rests on, pinned numerically."""
        logits, labels, tl, ul = _random_case(2, 8, 3, 6, seed=407)
        la, lb, lz = self._halves(logits, labels, tl, ul)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        b, t_len, u1, c = logits.shape
        lab = labels.clamp(min=0)
        gathered = torch.gather(
            log_probs[:, :, :u1 - 1, :], 3,
            lab.view(b, 1, u1 - 1, 1).expand(b, t_len, u1 - 1, 1)).squeeze(3)
        lpl = torch.cat(
            [gathered, torch.full((b, t_len, 1), rnnt_shc_loss.NEG_INF)], 2)

        def derive(a_, b_):
            qb, ql = rnnt_shc_loss.rnnt_transition_posteriors(
                a_, b_, log_probs[..., 0], lpl, lz, tl, ul)
            return rnnt_shc_loss._node_target(qb, ql, lab, c, 0)

        t0, g0 = derive(la, lb)
        la_s = rnnt_shc_loss.apply_alpha_beta_diagonal_smoothing(
            la, tl, ul, 0.05, 0.0, eps=1e-10)
        lb_s = rnnt_shc_loss.apply_alpha_beta_diagonal_smoothing(
            lb, tl, ul, 0.05, 0.0, eps=1e-10)
        t_a, g_a = derive(la_s, lb)
        t_b, g_b = derive(la, lb_s)
        self.assertLess(float((t_a - t0).abs().max()), 1e-5,
                        msg="smoothing alpha must not move z_hat")
        self.assertGreater(float((g_a - g0).abs().max()), 1e-4)
        self.assertGreater(float((t_b - t0).abs().max()), 1e-4,
                           msg="smoothing beta must move z_hat")

    def test_end_to_end(self):
        logits, labels, tl, ul = _random_case(2, 8, 3, 6, seed=409)
        grads = []
        for a in (0.0, 0.05):
            x = logits.clone().requires_grad_(True)
            loss = rnnt_shc_loss.rnnt_shc_loss(
                labels, ul, x, tl, blank=0, alpha=a,
                alpha_mode="aws_alpha_beta", fas_eps=1e-10)
            loss.sum().backward()
            grads.append((loss.detach().clone(), x.grad.clone()))
        self.assertTrue(torch.equal(grads[0][0], grads[1][0]),
                        msg="the loss is not a function of the smoothing")
        self.assertTrue(torch.isfinite(grads[1][1]).all())
        self.assertGreater(
            float((grads[0][1] - grads[1][1]).norm() / grads[0][1].norm()),
            1e-3)

    def test_nodes_outside_the_rectangle_get_no_gradient(self):
        logits, labels, tl, ul = _random_case(3, 10, 5, 6, seed=411,
                                              ragged=True)
        x = logits.clone().requires_grad_(True)
        loss = rnnt_shc_loss.rnnt_shc_loss(
            labels, ul, x, tl, blank=0, alpha=0.05,
            alpha_mode="aws_alpha_beta", fas_eps=1e-10)
        loss.sum().backward()
        for i, (t, u) in enumerate(zip(tl.tolist(), ul.tolist())):
            self.assertTrue(torch.allclose(
                x.grad[i, t:], torch.zeros_like(x.grad[i, t:]), atol=1e-9))
            self.assertTrue(torch.allclose(
                x.grad[i, :, u + 1:],
                torch.zeros_like(x.grad[i, :, u + 1:]), atol=1e-9))
