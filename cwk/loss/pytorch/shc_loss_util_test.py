"""Unit tests for the seq_loss_util module."""

# pylint: disable=import-error, no-member, no-name-in-module

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import os
import unittest
from cwk.loss.pytorch import shc_loss_util

# Third-party imports
import numpy as np
import torch

# Custom imports
from cwk.loss.pytorch import seq_loss_util

# Sets the log of minius inifinty of the float 32 type.
LOG_0 = seq_loss_util.LOG_0

"""Simple, intuitive unit tests for shc_loss_util.apply_post_processing.

NOTE: these tests use the "flipped" beta convention, where
new_beta = 1 - old_beta. In this convention, beta weights the
masked-uniform component u_p (via gamma = beta / p_p), and
(1 - beta) weights the plain-uniform component u.
"""


class ApplyPostProcessingTest(unittest.TestCase):
    """Tests for shc_loss_util.apply_post_processing correctness."""

    def test_single_step_manual_computation(self):
        """Checks one (b, t) cell against a hand-computed value.

        Distribution: [0.5, 0.5, 0.0, 0.0], C = 4, eps = 1e-10.
        Only 2 classes are "active" (prob >= eps).
        p_p = 2 / 4 = 0.5.
        u_p = [0.25, 0.25, 0, 0] (1/C on active classes).
        u = 1 / 4 = 0.25 everywhere.
        With alpha = 0.5, beta = 0.5:
            gamma = 0.5 / 0.5 = 1.0
            mix = (1-0.5) * 0.25 + 1.0 * u_p
                = 0.125 + [0.25, 0.25, 0, 0]
                = [0.375, 0.375, 0.125, 0.125]   (sums to 1)
            y_ls = 0.5 * y + 0.5 * mix
                 = 0.5*[0.5,0.5,0,0] + 0.5*[.375,.375,.125,.125]
                 = [0.4375, 0.4375, 0.0625, 0.0625]
        (beta = 0.5 is the symmetric point, so this value is the
        same under either beta convention.)
        """
        est_probs = torch.tensor([[[0.5, 0.5, 0.0, 0.0]]])
        logits_len = torch.tensor([1])

        out = shc_loss_util.apply_post_processing(
            est_probs, logits_len, alpha=0.5, beta=0.5, eps=1e-10)

        expected = torch.tensor([[[0.4375, 0.4375, 0.0625, 0.0625]]])
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_alpha_zero_is_identity_on_valid_steps(self):
        """alpha = 0 should leave valid (unpadded) steps unchanged.

        beta's value is irrelevant here since alpha = 0 discards
        the entire smoothing term.
        """
        torch.manual_seed(0)
        est_probs = torch.rand(2, 5, 7)
        est_probs = est_probs / est_probs.sum(dim=-1, keepdim=True)
        logits_len = torch.tensor([5, 5])

        out = shc_loss_util.apply_post_processing(
            est_probs, logits_len, alpha=0.0, beta=0.7, eps=1e-10)

        self.assertTrue(torch.allclose(out, est_probs, atol=1e-6))

    def test_padding_positions_are_zeroed(self):
        """Time steps t >= logits_len[b] must be all zeros."""
        est_probs = torch.rand(2, 6, 4)
        est_probs = est_probs / est_probs.sum(dim=-1, keepdim=True)
        logits_len = torch.tensor([3, 6])

        out = shc_loss_util.apply_post_processing(
            est_probs, logits_len, alpha=0.4, beta=0.4, eps=1e-10)

        # Batch 0: steps 3, 4, 5 are padding -> should be all zero.
        self.assertTrue(torch.all(out[0, 3:, :] == 0))
        # Batch 1: fully valid -> no forced zeros.
        self.assertFalse(torch.all(out[1] == 0))

    def test_output_shape_and_dtype_match_input(self):
        """Output shape/dtype must match est_probs exactly."""
        est_probs = torch.rand(3, 4, 6, dtype=torch.float32)
        est_probs = est_probs / est_probs.sum(dim=-1, keepdim=True)
        logits_len = torch.tensor([4, 2, 3])

        out = shc_loss_util.apply_post_processing(
            est_probs, logits_len, alpha=0.2, beta=0.9, eps=1e-10)

        self.assertEqual(out.shape, est_probs.shape)
        self.assertEqual(out.dtype, est_probs.dtype)

    def test_uniform_input_is_unchanged_by_smoothing(self):
        """If y is already uniform, smoothing should not change it.

        When every class prob equals 1/C (all >= eps), p_p = 1,
        so u_p = u = uniform and gamma = beta / p_p = beta. The
        mix becomes (1-beta)*u + beta*u = u for ANY beta, so
        y_ls = (1-alpha)*u + alpha*u = u.
        """
        c = 5
        est_probs = torch.full((1, 1, c), 1.0 / c)
        logits_len = torch.tensor([1])

        out = shc_loss_util.apply_post_processing(
            est_probs, logits_len, alpha=0.7, beta=0.6, eps=1e-10)

        self.assertTrue(torch.allclose(out, est_probs, atol=1e-6))

    def test_all_zero_beyond_eps_never_happens_but_handles_one_hot(self):
        """One-hot distribution: only one class is active.

        p_p = 1/C, so gamma = beta / p_p = beta * C, which
        concentrates the smoothing mass mostly back onto the
        single active class (since u_p is one-hot on it too).
        Sanity check: output stays a valid, non-negative vector
        and the originally-dominant class keeps the largest mass.
        """
        est_probs = torch.tensor([[[1.0, 0.0, 0.0]]])
        logits_len = torch.tensor([1])

        out = shc_loss_util.apply_post_processing(
            est_probs, logits_len, alpha=0.3, beta=0.5, eps=1e-6)

        self.assertTrue(torch.all(out >= 0))
        # Active class should retain the largest probability mass.
        self.assertGreater(out[0, 0, 0].item(), out[0, 0, 1].item())
        self.assertGreater(out[0, 0, 0].item(), out[0, 0, 2].item())

    def test_beta_zero_matches_standard_label_smoothing(self):
        """beta = 0 must reduce to standard label smoothing.

        With beta = 0, gamma = 0 / p_p = 0, so the mix collapses
        to plain uniform u regardless of p_p. This should equal
        (1 - alpha) * y + alpha * u for ANY input, including one
        with very sparse (mostly-zero) rows. (This is the
        beta = 1 case from the old, pre-flip convention.)
        """
        c = 4
        est_probs = torch.tensor([[[0.9, 0.1, 0.0, 0.0]]])
        logits_len = torch.tensor([1])
        alpha = 0.6

        out = shc_loss_util.apply_post_processing(
            est_probs, logits_len, alpha=alpha, beta=0.0, eps=1e-10)

        u = torch.full_like(est_probs, 1.0 / c)
        expected = (1 - alpha) * est_probs + alpha * u
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_beta_one_with_full_support_matches_standard_ls(self):
        """beta = 1 with p_p = 1 must also match standard LS.

        p_p = 1 requires every class prob >= eps. In that case
        u_p is itself the uniform distribution u, and
        gamma = beta / p_p = 1, so gamma * u_p = u. The (1-beta)
        term vanishes, and the result again equals
        (1 - alpha) * y + alpha * u. (This is the beta = 0 case
        from the old, pre-flip convention.)
        """
        c = 4
        # All entries are comfortably above eps -> p_p = 1.
        est_probs = torch.tensor([[[0.7, 0.1, 0.1, 0.1]]])
        logits_len = torch.tensor([1])
        alpha = 0.35

        out = shc_loss_util.apply_post_processing(
            est_probs, logits_len, alpha=alpha, beta=1.0, eps=1e-10)

        u = torch.full_like(est_probs, 1.0 / c)
        expected = (1 - alpha) * est_probs + alpha * u
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_batch_with_different_lengths(self):
        """Examples in a batch may have different valid lengths.

        Shape is (B=2, T=3, C=3). Example 0 has length 3 (fully
        valid); example 1 has length 2 (last step is padding).
        Each (b, t) cell is smoothed independently (unlike plain
        LS, there is no averaging across time steps here) using:
            p_p   = (# classes with prob >= eps) / C
            u_p   = 1/C on active classes, 0 elsewhere
            gamma = beta / p_p
            mix   = (1-beta) * u + gamma * u_p, with u = 1/C
                    (mix sums to 1, since (1-beta)+gamma*p_p = 1)
            out   = (1 - alpha) * y + alpha * mix
        Padded time steps (t >= logits_len[b]) are zeroed out.
        Uses alpha=0.4, beta=0.7, eps=1e-6 (the new-convention
        equivalent of the old beta=0.3).
        """
        est_probs = torch.tensor([
            [
                [0.6, 0.4, 0.0],
                [0.5, 0.5, 0.0],
                [0.7, 0.3, 0.0],
            ],
            [
                [0.5, 0.3, 0.2],
                [0.9, 0.1, 0.0],
                [0.9, 0.05, 0.05],  # padding, ignore.
            ],
        ])
        logits_len = torch.tensor([3, 2])

        out = shc_loss_util.apply_post_processing(
            est_probs, logits_len, alpha=0.4, beta=0.7, eps=1e-6)

        # ex0: every row has 2 active classes (idx 0, 1) -> p_p =
        #   2/3, gamma = 0.7/(2/3) = 1.05. u_p = [1/3, 1/3, 0].
        #   mix = (1-0.7)*(1/3) + gamma*u_p
        #       = [0.1+0.35, 0.1+0.35, 0.1+0] = [0.45, 0.45, 0.1]
        #       for every row (mix only depends on p_p, which is
        #       the same for all 3 rows here). Sums to 1.
        #   row0: 0.6*[0.6,0.4,0.0] + 0.4*mix = [0.54, 0.42, 0.04]
        #   row1: 0.6*[0.5,0.5,0.0] + 0.4*mix = [0.48, 0.48, 0.04]
        #   row2: 0.6*[0.7,0.3,0.0] + 0.4*mix = [0.6, 0.36, 0.04]
        #
        # ex1: only rows 0, 1 are valid.
        #   row0: all 3 classes active -> p_p = 1, gamma = 0.7,
        #       u_p = u -> mix = (1-beta+gamma)*u = 1*(1/3).
        #       out = 0.6*[0.5,0.3,0.2] + 0.4*[1/3,1/3,1/3]
        #           = [0.433333, 0.313333, 0.253333]
        #   row1: 2 active classes -> p_p = 2/3, same mix as ex0
        #       rows = [0.45, 0.45, 0.1].
        #       out = 0.6*[0.9,0.1,0.0] + 0.4*mix
        #           = [0.72, 0.24, 0.04]
        #   row2: padding -> forced to all zeros.
        expected_output = torch.tensor([
            [
                [0.54, 0.42, 0.04],
                [0.48, 0.48, 0.04],
                [0.6, 0.36, 0.04],
            ],
            [
                [0.433333, 0.313333, 0.253333],
                [0.72, 0.24, 0.04],
                [0.0, 0.0, 0.0],
            ],
        ])
        self.assertTrue(
            torch.allclose(out, expected_output, atol=1e-5))

    def test_output_sums_to_one_on_valid_steps(self):
        """Smoothed distribution must sum to 1 on valid steps.

        This is the property that requires gamma = beta / p_p
        instead of beta * p_p: since sum(u) = 1 and
        sum(u_p) = p_p, only the division keeps
        ((1-beta)*u + gamma*u_p), and hence y_ls, a valid
        probability distribution over classes for every (b, t).
        """
        torch.manual_seed(1)
        est_probs = torch.rand(2, 4, 6)
        # Zero out a random subset of classes per row, then
        # renormalize, so p_p varies across rows (not all 1.0).
        zero_mask = torch.rand(2, 4, 6) < 0.4
        est_probs = est_probs.masked_fill(zero_mask, 0.0)
        # Guarantee at least one active class per row.
        est_probs[..., 0] = est_probs[..., 0].clamp(min=0.1)
        est_probs = est_probs / est_probs.sum(dim=-1, keepdim=True)
        logits_len = torch.tensor([4, 3])

        out = shc_loss_util.apply_post_processing(
            est_probs, logits_len, alpha=0.5, beta=0.8, eps=1e-6)

        sums = out.sum(dim=-1)  # (B, T)
        # Valid steps must sum to 1.
        self.assertTrue(torch.allclose(
            sums[0, :4], torch.ones(4), atol=1e-5))
        self.assertTrue(torch.allclose(
            sums[1, :3], torch.ones(3), atol=1e-5))
        # Padded step (batch 1, t=3) was zeroed out, so sums to 0.
        self.assertAlmostEqual(sums[1, 3].item(), 0.0, places=5)


class ApplyPeakPreservingSelectiveEstimatedTargetSmoothingTest(
        unittest.TestCase):
    """Tests for apply_peak_preserving_selective_estimated_target_smoothing.

    Formula under test, for each (b, t), with peak = argmax(y),
    S_rest = 1 - y[peak], A' = {c != peak : y[c] >= eps}, N' = |A'|:
        y_new[peak]                    = y[peak]
        y_new[c], c in A'              = (1-gamma)*y[c]
                                          + gamma * S_rest / N'
        y_new[c], c not in A', c!=peak = (1-gamma)*y[c]
    If N' == 0, y_new = y unchanged for that (b, t).
    """

    def test_single_step_manual_computation(self):
        """Checks one (b, t) cell against a hand-computed value.

        y = [0.5, 0.3, 0.2, 0.0], eps = 1e-6, gamma = 0.4.
        peak = 0 (0.5). S_rest = 0.5. A' = {1, 2} (0.3, 0.2 >= eps);
        index 3 (0.0) is below eps, so it's scaled but not
        redistributed onto. N' = 2.
            y_new[0] = 0.5                              (peak, unchanged)
            y_new[1] = 0.6*0.3 + 0.4*0.5/2 = 0.18+0.10 = 0.28
            y_new[2] = 0.6*0.2 + 0.4*0.5/2 = 0.12+0.10 = 0.22
            y_new[3] = 0.6*0.0                        = 0.00
        """
        est_probs = torch.tensor([[[0.5, 0.3, 0.2, 0.0]]])
        logits_len = torch.tensor([1])

        out = shc_loss_util.\
            apply_peak_preserving_selective_estimated_target_smoothing(
                est_probs, logits_len, gamma=0.4, eps=1e-6)

        expected = torch.tensor([[[0.5, 0.28, 0.22, 0.0]]])
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_peak_class_is_always_unchanged(self):
        """The argmax class must keep its exact original probability,
        regardless of gamma."""
        torch.manual_seed(0)
        est_probs = torch.rand(2, 5, 7)
        est_probs = est_probs / est_probs.sum(dim=-1, keepdim=True)
        logits_len = torch.tensor([5, 5])

        out = shc_loss_util.\
            apply_peak_preserving_selective_estimated_target_smoothing(
                est_probs, logits_len, gamma=0.9, eps=1e-6)

        peak_idx = est_probs.argmax(dim=-1)
        out_peak = torch.gather(out, -1, peak_idx.unsqueeze(-1))
        in_peak = torch.gather(est_probs, -1, peak_idx.unsqueeze(-1))
        self.assertTrue(torch.allclose(out_peak, in_peak, atol=1e-6))

    def test_gamma_zero_is_identity_on_valid_steps(self):
        """gamma = 0 should leave valid (unpadded) steps unchanged."""
        torch.manual_seed(1)
        est_probs = torch.rand(2, 4, 6)
        est_probs = est_probs / est_probs.sum(dim=-1, keepdim=True)
        logits_len = torch.tensor([4, 4])

        out = shc_loss_util.\
            apply_peak_preserving_selective_estimated_target_smoothing(
                est_probs, logits_len, gamma=0.0, eps=1e-6)

        self.assertTrue(torch.allclose(out, est_probs, atol=1e-6))

    def test_one_hot_case_has_nothing_to_redistribute_onto(self):
        """If only the peak class is active (N' = 0), the output must
        equal the input exactly -- there is no other active class to
        move mass onto."""
        est_probs = torch.tensor([[[1.0, 0.0, 0.0]]])
        logits_len = torch.tensor([1])

        out = shc_loss_util.\
            apply_peak_preserving_selective_estimated_target_smoothing(
                est_probs, logits_len, gamma=0.5, eps=1e-6)

        self.assertTrue(torch.allclose(out, est_probs, atol=1e-6))

    def test_output_sums_to_one_on_valid_steps(self):
        """Smoothed distribution must sum to 1 on valid steps,
        independent of gamma (see the docstring derivation)."""
        torch.manual_seed(2)
        est_probs = torch.rand(2, 4, 6)
        zero_mask = torch.rand(2, 4, 6) < 0.4
        est_probs = est_probs.masked_fill(zero_mask, 0.0)
        # Guarantee at least one active non-peak class per row so
        # N' > 0 (keeps this test focused on the N' > 0 branch;
        # the N' == 0 branch is covered separately above).
        est_probs[..., 0] = est_probs[..., 0].clamp(min=0.2)
        est_probs[..., 1] = est_probs[..., 1].clamp(min=0.1)
        est_probs = est_probs / est_probs.sum(dim=-1, keepdim=True)
        logits_len = torch.tensor([4, 3])

        out = shc_loss_util.\
            apply_peak_preserving_selective_estimated_target_smoothing(
                est_probs, logits_len, gamma=0.6, eps=1e-6)

        sums = out.sum(dim=-1)  # (B, T)
        self.assertTrue(torch.allclose(
            sums[0, :4], torch.ones(4), atol=1e-5))
        self.assertTrue(torch.allclose(
            sums[1, :3], torch.ones(3), atol=1e-5))
        # Padded step (batch 1, t=3) was zeroed out.
        self.assertAlmostEqual(sums[1, 3].item(), 0.0, places=5)

    def test_padding_positions_are_zeroed(self):
        """Time steps t >= logits_len[b] must be all zeros."""
        est_probs = torch.rand(2, 6, 4)
        est_probs = est_probs / est_probs.sum(dim=-1, keepdim=True)
        logits_len = torch.tensor([3, 6])

        out = shc_loss_util.\
            apply_peak_preserving_selective_estimated_target_smoothing(
                est_probs, logits_len, gamma=0.4, eps=1e-6)

        self.assertTrue(torch.all(out[0, 3:, :] == 0))
        self.assertFalse(torch.all(out[1] == 0))

    def test_output_shape_and_dtype_match_input(self):
        """Output shape/dtype must match est_probs exactly."""
        est_probs = torch.rand(3, 4, 6, dtype=torch.float32)
        est_probs = est_probs / est_probs.sum(dim=-1, keepdim=True)
        logits_len = torch.tensor([4, 2, 3])

        out = shc_loss_util.\
            apply_peak_preserving_selective_estimated_target_smoothing(
                est_probs, logits_len, gamma=0.3, eps=1e-6)

        self.assertEqual(out.shape, est_probs.shape)
        self.assertEqual(out.dtype, est_probs.dtype)

    def test_batch_with_different_lengths(self):
        """Examples in a batch may have different valid lengths, and
        N' may differ per (b, t) (here 2 active non-peak classes in
        batch 0, 1 active non-peak class in batch 1). gamma = 0.5,
        eps = 1e-6.

        batch 0, row 0: [0.7, 0.2, 0.1]. peak=0. S_rest=0.3.
            A'={1,2}, N'=2.
            -> [0.7, 0.5*0.2+0.5*0.3/2, 0.5*0.1+0.5*0.3/2]
             = [0.7, 0.175, 0.125]
        batch 0, row 1: [0.5, 0.3, 0.2]. peak=0. S_rest=0.5.
            A'={1,2}, N'=2.
            -> [0.5, 0.5*0.3+0.5*0.5/2, 0.5*0.2+0.5*0.5/2]
             = [0.5, 0.275, 0.225]
        batch 1, row 0: [0.9, 0.1, 0.0]. peak=0. S_rest=0.1.
            A'={1} (index 2 is below eps), N'=1.
            -> [0.9, 0.5*0.1+0.5*0.1/1, 0.5*0.0]
             = [0.9, 0.1, 0.0]
        batch 1, row 1: padding -> zeroed regardless of value.
        """
        est_probs = torch.tensor([
            [
                [0.7, 0.2, 0.1],
                [0.5, 0.3, 0.2],
            ],
            [
                [0.9, 0.1, 0.0],
                [0.9, 0.05, 0.05],  # padding, ignored.
            ],
        ])
        logits_len = torch.tensor([2, 1])

        out = shc_loss_util.\
            apply_peak_preserving_selective_estimated_target_smoothing(
                est_probs, logits_len, gamma=0.5, eps=1e-6)

        expected_output = torch.tensor([
            [
                [0.7, 0.175, 0.125],
                [0.5, 0.275, 0.225],
            ],
            [
                [0.9, 0.1, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ])
        self.assertTrue(torch.allclose(out, expected_output, atol=1e-5))

    def test_dispatcher_routes_to_peak_preserving_variant(self):
        """apply_post_processing(peak_preserving=True, gamma=...) must
        match calling the peak-preserving function directly, and must
        ignore alpha/beta entirely."""
        torch.manual_seed(3)
        est_probs = torch.rand(2, 3, 5)
        est_probs = est_probs / est_probs.sum(dim=-1, keepdim=True)
        logits_len = torch.tensor([3, 2])

        direct = shc_loss_util.\
            apply_peak_preserving_selective_estimated_target_smoothing(
                est_probs, logits_len, gamma=0.4, eps=1e-6)
        via_dispatcher = shc_loss_util.apply_post_processing(
            est_probs, logits_len, alpha=0.123, beta=0.456, eps=1e-6,
            peak_preserving=True, gamma=0.4)

        self.assertTrue(torch.allclose(direct, via_dispatcher, atol=1e-6))


class ApplyPeakCappingSelectiveEstimatedTargetSmoothingTest(unittest.TestCase):
    """Tests for apply_peak_capping_selective_estimated_target_smoothing.

    Formula under test, for each (b, t), with peak = argmax(y):
        if y[peak] > (1 - alpha):
            k = (1 - alpha) / y[peak]
            scaled = k * y
            lost_mass = 1 - k
            A' = {c != peak : y[c] >= eps}, N' = |A'|
            y_new[peak]                    = scaled[peak]  (= 1-alpha)
            y_new[c], c in A'              = scaled[c] + lost_mass/N'
            y_new[c], c not in A', c!=peak = scaled[c]
        else:
            y_new = y
    """

    def test_single_step_manual_computation(self):
        """Checks one (b, t) cell against a hand-computed value.

        y = [0.9, 0.06, 0.04, 0.0], eps = 1e-6, alpha = 0.15 ->
        cap = 0.85. peak = 0 (0.9 > 0.85, cap exceeded).
            k = 0.85 / 0.9 = 17/18
            scaled = [0.85, 0.06*17/18, 0.04*17/18, 0]
                   = [0.85, 0.056667, 0.037778, 0.0]
            lost_mass = 1/18 = 0.055556
            A' = {1, 2} (index 3 is below eps), N' = 2
            per_class_share = (1/18)/2 = 1/36 = 0.027778
            y_new = [0.85, 0.084444, 0.065556, 0.0]
        """
        est_probs = torch.tensor([[[0.9, 0.06, 0.04, 0.0]]])
        logits_len = torch.tensor([1])

        out = shc_loss_util.\
            apply_peak_capping_selective_estimated_target_smoothing(
                est_probs, logits_len, alpha=0.15, eps=1e-6)

        expected = torch.tensor([[[0.85, 0.084444, 0.065556, 0.0]]])
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_below_cap_is_untouched(self):
        """If the peak is already at or below (1 - alpha), the row must
        be left completely unchanged."""
        est_probs = torch.tensor([[[0.9, 0.06, 0.04, 0.0]]])
        logits_len = torch.tensor([1])

        # cap = 1 - 0.2 = 0.8 < 0.9 would trigger; use alpha small
        # enough that cap = 1 - alpha >= 0.9 so the row is untouched.
        out = shc_loss_util.\
            apply_peak_capping_selective_estimated_target_smoothing(
                est_probs, logits_len, alpha=0.05, eps=1e-6)

        self.assertTrue(torch.allclose(out, est_probs, atol=1e-6))

    def test_peak_class_hits_cap_exactly_when_triggered(self):
        """When the cap is exceeded, the peak class must land exactly
        on (1 - alpha), regardless of alpha/starting peak value."""
        torch.manual_seed(4)
        est_probs = torch.rand(2, 5, 7)
        est_probs = est_probs / est_probs.sum(dim=-1, keepdim=True)
        logits_len = torch.tensor([5, 5])
        alpha = 0.5  # aggressive cap so it's likely triggered widely.

        out = shc_loss_util.\
            apply_peak_capping_selective_estimated_target_smoothing(
                est_probs, logits_len, alpha=alpha, eps=1e-6)

        peak_idx = est_probs.argmax(dim=-1)
        in_peak = torch.gather(est_probs, -1, peak_idx.unsqueeze(-1))
        out_peak = torch.gather(out, -1, peak_idx.unsqueeze(-1))
        triggered = in_peak > (1 - alpha)
        # Wherever triggered, out_peak must equal (1 - alpha) exactly.
        expected_capped = torch.full_like(out_peak, 1 - alpha)
        self.assertTrue(torch.allclose(
            out_peak[triggered], expected_capped[triggered], atol=1e-5))
        # Wherever not triggered, out_peak must equal the original.
        self.assertTrue(torch.allclose(
            out_peak[~triggered], in_peak[~triggered], atol=1e-6))

    def test_one_hot_case_has_nothing_to_redistribute_onto(self):
        """If only the peak class is active (N' = 0), the output must
        equal the input exactly even though the cap is exceeded."""
        est_probs = torch.tensor([[[1.0, 0.0, 0.0]]])
        logits_len = torch.tensor([1])

        out = shc_loss_util.\
            apply_peak_capping_selective_estimated_target_smoothing(
                est_probs, logits_len, alpha=0.5, eps=1e-6)

        self.assertTrue(torch.allclose(out, est_probs, atol=1e-6))

    def test_output_sums_to_one_on_valid_steps(self):
        """Smoothed distribution must sum to 1 on valid steps, whether
        or not the cap was triggered for a given (b, t)."""
        torch.manual_seed(5)
        est_probs = torch.rand(2, 4, 6)
        zero_mask = torch.rand(2, 4, 6) < 0.4
        est_probs = est_probs.masked_fill(zero_mask, 0.0)
        est_probs[..., 0] = est_probs[..., 0].clamp(min=0.2)
        est_probs[..., 1] = est_probs[..., 1].clamp(min=0.1)
        est_probs = est_probs / est_probs.sum(dim=-1, keepdim=True)
        logits_len = torch.tensor([4, 3])

        out = shc_loss_util.\
            apply_peak_capping_selective_estimated_target_smoothing(
                est_probs, logits_len, alpha=0.3, eps=1e-6)

        sums = out.sum(dim=-1)  # (B, T)
        self.assertTrue(torch.allclose(
            sums[0, :4], torch.ones(4), atol=1e-5))
        self.assertTrue(torch.allclose(
            sums[1, :3], torch.ones(3), atol=1e-5))
        self.assertAlmostEqual(sums[1, 3].item(), 0.0, places=5)

    def test_padding_positions_are_zeroed(self):
        """Time steps t >= logits_len[b] must be all zeros."""
        est_probs = torch.rand(2, 6, 4)
        est_probs = est_probs / est_probs.sum(dim=-1, keepdim=True)
        logits_len = torch.tensor([3, 6])

        out = shc_loss_util.\
            apply_peak_capping_selective_estimated_target_smoothing(
                est_probs, logits_len, alpha=0.3, eps=1e-6)

        self.assertTrue(torch.all(out[0, 3:, :] == 0))
        self.assertFalse(torch.all(out[1] == 0))

    def test_output_shape_and_dtype_match_input(self):
        """Output shape/dtype must match est_probs exactly."""
        est_probs = torch.rand(3, 4, 6, dtype=torch.float32)
        est_probs = est_probs / est_probs.sum(dim=-1, keepdim=True)
        logits_len = torch.tensor([4, 2, 3])

        out = shc_loss_util.\
            apply_peak_capping_selective_estimated_target_smoothing(
                est_probs, logits_len, alpha=0.2, eps=1e-6)

        self.assertEqual(out.shape, est_probs.shape)
        self.assertEqual(out.dtype, est_probs.dtype)

    def test_batch_with_different_lengths(self):
        """Examples in a batch may have different valid lengths.

        batch 0, row 0: [0.9, 0.06, 0.04]. alpha=0.15 -> cap=0.85.
            peak=0.9 > 0.85 triggered. k=0.85/0.9=17/18.
            scaled=[0.85, 0.056667, 0.037778]. lost_mass=1/18.
            A'={1,2}, N'=2, share=1/36=0.027778.
            -> [0.85, 0.084444, 0.065556]
        batch 0, row 1: [0.7, 0.2, 0.1]. alpha=0.15 -> cap=0.85.
            peak=0.7 <= 0.85, not triggered -> unchanged.
        batch 1, row 0: [0.95, 0.05, 0.0]. alpha=0.15 -> cap=0.85.
            peak=0.95 > 0.85 triggered. k=0.85/0.95=17/19.
            scaled=[0.85, 0.05*17/19, 0]=[0.85, 0.044737, 0.0].
            lost_mass=2/19=0.105263.
            A'={1} (index 2 below eps), N'=1.
            -> [0.85, 0.044737+0.105263, 0.0] = [0.85, 0.15, 0.0]
        batch 1, row 1: padding -> zeroed regardless of value.
        """
        est_probs = torch.tensor([
            [
                [0.9, 0.06, 0.04],
                [0.7, 0.2, 0.1],
            ],
            [
                [0.95, 0.05, 0.0],
                [0.9, 0.05, 0.05],  # padding, ignored.
            ],
        ])
        logits_len = torch.tensor([2, 1])

        out = shc_loss_util.\
            apply_peak_capping_selective_estimated_target_smoothing(
                est_probs, logits_len, alpha=0.15, eps=1e-6)

        expected_output = torch.tensor([
            [
                [0.85, 0.084444, 0.065556],
                [0.7, 0.2, 0.1],
            ],
            [
                [0.85, 0.15, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ])
        self.assertTrue(torch.allclose(out, expected_output, atol=1e-5))

    def test_dispatcher_routes_to_peak_capping_variant(self):
        """apply_post_processing(peak_capping=True) must match calling
        the peak-capping function directly, and must ignore beta/gamma
        entirely."""
        torch.manual_seed(6)
        est_probs = torch.rand(2, 3, 5)
        est_probs = est_probs / est_probs.sum(dim=-1, keepdim=True)
        logits_len = torch.tensor([3, 2])

        direct = shc_loss_util.\
            apply_peak_capping_selective_estimated_target_smoothing(
                est_probs, logits_len, alpha=0.3, eps=1e-6)
        via_dispatcher = shc_loss_util.apply_post_processing(
            est_probs, logits_len, alpha=0.3, beta=0.789, eps=1e-6,
            peak_capping=True, gamma=0.111)

        self.assertTrue(torch.allclose(direct, via_dispatcher, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
