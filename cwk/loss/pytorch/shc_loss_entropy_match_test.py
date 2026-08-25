"""Unit tests for shc_loss_util.apply_entropy_matched_smoothing.

The function solves, per example, for the alpha that makes the smoothed
target's mean per-frame entropy equal the acoustic posterior's. Most of
what can go wrong is either (a) the solve landing on the wrong value or
(b) one of the cases where no solution exists being handled silently and
incorrectly, so the tests below pin down both, plus the structural claim
the solve relies on:

    h(a) = H((1-a) q~ + a m) is concave with h'(1) = 0, hence
    non-decreasing on [0, 1], hence bisection converges to the unique root.

test_entropy_is_monotone_in_alpha and test_h_prime_1_is_zero check that
claim numerically rather than taking it on faith, since the whole
algorithm is invalid without it.
"""

# pylint: disable=import-error, no-member, no-name-in-module

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import unittest

# Third-party imports
import torch

# Custom imports
from cwk.loss.pytorch import shc_loss_util


def _entropy(probs):
    """Shannon entropy (nats) over the last axis."""
    return -torch.xlogy(probs, probs).sum(dim=-1)


def _mean_frame_entropy(probs, logits_len):
    """Mean per-frame entropy over each example's valid frames."""
    per_frame = _entropy(probs)
    time_idx = torch.arange(probs.shape[1])
    valid = (time_idx.unsqueeze(0) < logits_len.unsqueeze(1)).float()
    return (per_frame * valid).sum(dim=1) / logits_len.float()


def _interior_case(batch=3, max_time=4):
    """A hand-built case whose solution lands strictly inside (0, 1).

    Randomly generated cases mostly clamp: a random target over a small
    active set is already near-uniform on it, so its ceiling log N sits
    below the entropy of a random softmax over the full vocabulary, and
    every example lands in case B. That is realistic (it is what early
    training looks like) but it exercises no bisection, so the tests that
    are actually about the solve use this fixture instead.

    Target row: 6 active classes, H(q~) ~= 1.06 nats, ceiling
    log 6 ~= 1.79. Acoustic row: H(p) ~= 1.66, comfortably between them.
    """
    target_row = torch.tensor(
        [0.70, 0.10, 0.08, 0.06, 0.04, 0.02, 0.0, 0.0])
    acoustic_row = torch.tensor(
        [0.40, 0.20, 0.15, 0.10, 0.07, 0.05, 0.02, 0.01])

    target = target_row.repeat(batch, max_time, 1).clone()
    acoustic = torch.empty(batch, max_time, acoustic_row.shape[0])
    for b in range(batch):
        # Sharpen slightly per example so the solved alphas differ.
        sharpened = torch.softmax(
            torch.log(acoustic_row) * (1.0 + 0.15 * b), dim=-1)
        acoustic[b] = sharpened.repeat(max_time, 1)

    logits_len = torch.full((batch,), max_time)
    return target, acoustic, logits_len


def _random_case(seed=0, batch=4, max_time=6, num_classes=8, sparsity=0.5):
    """A target/acoustic pair with a realistically sparse active set.

    `sparsity` is the fraction of classes forced to exactly zero in the
    target, mimicking classes absent from the transcript.
    """
    torch.manual_seed(seed)
    target = torch.rand(batch, max_time, num_classes)
    drop = torch.rand(batch, max_time, num_classes) < sparsity
    # Keep at least one class alive per frame.
    drop[..., 0] = False
    target = target * (~drop)
    target = target / target.sum(dim=-1, keepdim=True)
    acoustic = torch.softmax(
        torch.randn(batch, max_time, num_classes), dim=-1)
    logits_len = torch.tensor([max_time, max_time - 1, max_time - 2,
                               max_time - 3][:batch])
    return target, acoustic, logits_len


class EntropyMatchedSmoothingTest(unittest.TestCase):
    """Tests for apply_entropy_matched_smoothing."""

    def test_entropy_is_monotone_in_alpha(self):
        """h(a) must be non-decreasing on [0, 1] -- bisection needs it."""
        target, _, logits_len = _random_case()
        eps = 1e-6
        active = target >= eps
        n_active = active.sum(dim=-1, keepdim=True).clamp(min=1).float()
        masked_uniform = active.float() / n_active
        restricted = target * active
        restricted = restricted / restricted.sum(dim=-1, keepdim=True)

        previous = None
        for step in range(41):
            a = step / 40.0
            mixture = (1.0 - a) * restricted + a * masked_uniform
            current = _mean_frame_entropy(mixture, logits_len)
            if previous is not None:
                # Allow only floating-point-scale decreases.
                self.assertTrue(
                    torch.all(current - previous > -1e-6),
                    f"entropy decreased at alpha={a}")
            previous = current

    def test_h_prime_1_is_zero(self):
        """The derivative at alpha=1 must vanish for uniform m."""
        target, acoustic, logits_len = _random_case(seed=1)
        _, stats = shc_loss_util.apply_entropy_matched_smoothing(
            target, acoustic, logits_len, return_stats=True)
        self.assertTrue(torch.all(stats["h_prime_1"].abs() < 1e-5))

    def test_solved_alpha_actually_matches_the_entropy(self):
        """The returned target's entropy must equal the acoustic one."""
        target, acoustic, logits_len = _interior_case()
        smoothed, stats = shc_loss_util.apply_entropy_matched_smoothing(
            target, acoustic, logits_len, return_stats=True)

        interior = (stats["case_b"] == 0) & (stats["case_c"] == 0)
        self.assertTrue(torch.all(interior), "fixture should not clamp")
        # A genuine solve, not a clamp parked at an endpoint.
        self.assertTrue(torch.all(stats["alpha"] > 1e-3))
        self.assertTrue(torch.all(stats["alpha"] < 1.0 - 1e-3))

        achieved = _mean_frame_entropy(smoothed, logits_len)
        self.assertTrue(torch.allclose(
            achieved, stats["h_target"], atol=1e-4))

    def test_case_b_clamps_to_one(self):
        """An unreachable target entropy must clamp at alpha = 1."""
        # A near-one-hot target has a tiny active set, so log N is small;
        # a uniform acoustic posterior demands far more entropy than that.
        target = torch.tensor([[[0.97, 0.01, 0.01, 0.01]]])
        acoustic = torch.full((1, 1, 4), 0.25)
        logits_len = torch.tensor([1])

        _, stats = shc_loss_util.apply_entropy_matched_smoothing(
            target, acoustic, logits_len, return_stats=True)

        self.assertEqual(stats["case_b"].item(), 1.0)
        self.assertAlmostEqual(stats["alpha"].item(), 1.0, places=6)

    def test_case_c_clamps_to_zero(self):
        """A target already more diffuse than the model must be left alone."""
        target = torch.full((1, 1, 4), 0.25)
        acoustic = torch.tensor([[[0.97, 0.01, 0.01, 0.01]]])
        logits_len = torch.tensor([1])

        smoothed, stats = shc_loss_util.apply_entropy_matched_smoothing(
            target, acoustic, logits_len, return_stats=True)

        self.assertEqual(stats["case_c"].item(), 1.0)
        self.assertAlmostEqual(stats["alpha"].item(), 0.0, places=6)
        self.assertTrue(torch.allclose(smoothed, target, atol=1e-6))

    def test_single_active_class_is_a_no_op(self):
        """N == 1 makes m == q~, so alpha is meaningless; leave y alone."""
        target = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        acoustic = torch.full((1, 1, 4), 0.25)
        logits_len = torch.tensor([1])

        smoothed, stats = shc_loss_util.apply_entropy_matched_smoothing(
            target, acoustic, logits_len, return_stats=True)

        self.assertAlmostEqual(stats["alpha"].item(), 0.0, places=6)
        self.assertTrue(torch.allclose(smoothed, target, atol=1e-6))

    def test_output_is_a_distribution_on_valid_frames(self):
        """Rows must stay non-negative and sum to 1 where they matter."""
        target, acoustic, logits_len = _random_case(seed=3)
        smoothed = shc_loss_util.apply_entropy_matched_smoothing(
            target, acoustic, logits_len)

        time_idx = torch.arange(smoothed.shape[1])
        valid = time_idx.unsqueeze(0) < logits_len.unsqueeze(1)
        sums = smoothed.sum(dim=-1)[valid]
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums),
                                       atol=1e-5))
        self.assertTrue(torch.all(smoothed >= 0.0))

    def test_padding_is_zeroed(self):
        """Padded frames must be zeroed, as in the other variants."""
        target, acoustic, logits_len = _random_case(seed=4)
        smoothed = shc_loss_util.apply_entropy_matched_smoothing(
            target, acoustic, logits_len)

        time_idx = torch.arange(smoothed.shape[1])
        padded = time_idx.unsqueeze(0) >= logits_len.unsqueeze(1)
        self.assertTrue(torch.all(smoothed[padded] == 0.0))

    def test_never_puts_mass_on_inactive_classes(self):
        """Classes absent from the target must stay at exactly zero."""
        target, acoustic, logits_len = _random_case(seed=5, sparsity=0.6)
        smoothed = shc_loss_util.apply_entropy_matched_smoothing(
            target, acoustic, logits_len)

        inactive = target < 1e-6
        time_idx = torch.arange(smoothed.shape[1])
        valid = (time_idx.unsqueeze(0) <
                 logits_len.unsqueeze(1)).unsqueeze(-1)
        self.assertTrue(torch.all(smoothed[inactive & valid] == 0.0))

    def test_alpha_max_caps_the_solution(self):
        """alpha_max must bind without disturbing anything else."""
        target, acoustic, logits_len = _random_case(seed=6)
        _, uncapped = shc_loss_util.apply_entropy_matched_smoothing(
            target, acoustic, logits_len, return_stats=True)
        _, capped = shc_loss_util.apply_entropy_matched_smoothing(
            target, acoustic, logits_len, alpha_max=0.05,
            return_stats=True)

        self.assertTrue(torch.all(capped["alpha"] <= 0.05 + 1e-6))
        self.assertTrue(torch.any(uncapped["alpha"] > 0.05),
                        "test case never exceeded the cap")

    def test_kappa_interpolates_the_target(self):
        """kappa scales the entropy gap, so alpha must shrink with it."""
        target, acoustic, logits_len = _interior_case()
        _, full = shc_loss_util.apply_entropy_matched_smoothing(
            target, acoustic, logits_len, kappa=1.0, return_stats=True)
        _, half = shc_loss_util.apply_entropy_matched_smoothing(
            target, acoustic, logits_len, kappa=0.5, return_stats=True)

        self.assertTrue(torch.all(half["alpha"] < full["alpha"]))
        # kappa only rescales the target entropy, so the achieved entropy
        # must sit exactly halfway up the gap.
        expected = full["h_lo"] + 0.5 * (full["h_target"] - full["h_lo"])
        self.assertTrue(torch.allclose(half["h_target"], expected,
                                       atol=1e-5))

    def test_rejects_mismatched_class_axes(self):
        """Label-space misuse must fail loudly, not silently misbehave."""
        target = torch.softmax(torch.randn(2, 3, 9), dim=-1)
        acoustic = torch.softmax(torch.randn(2, 3, 5), dim=-1)
        with self.assertRaises(AssertionError):
            shc_loss_util.apply_entropy_matched_smoothing(
                target, acoustic, torch.tensor([3, 3]))

    def test_more_iterations_do_not_move_the_answer(self):
        """20 bisection steps must already be converged."""
        target, acoustic, logits_len = _random_case(seed=8)
        _, coarse = shc_loss_util.apply_entropy_matched_smoothing(
            target, acoustic, logits_len, n_iter=20, return_stats=True)
        _, fine = shc_loss_util.apply_entropy_matched_smoothing(
            target, acoustic, logits_len, n_iter=40, return_stats=True)

        self.assertTrue(torch.allclose(coarse["alpha"], fine["alpha"],
                                       atol=1e-5))


if __name__ == "__main__":
    unittest.main()
