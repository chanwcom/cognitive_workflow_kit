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


if __name__ == "__main__":
    unittest.main()
