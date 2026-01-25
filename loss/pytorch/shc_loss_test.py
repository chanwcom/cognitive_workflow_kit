"""Unit tests for the shc_loss module."""

# pylint: disable=import-error, no-member, no-name-in-module

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import os
import unittest

# Third-party imports
import numpy as np
import torch

# Custom imports
from loss.pytorch import seq_loss_util
from loss.pytorch import shc_loss

class TestShiftOps(unittest.TestCase):
    """Unit tests for horizontal shift operations."""

    def setUp(self):
        self.x = torch.tensor([[1, 2, 3], 
                               [4, 5, 6]], dtype=torch.float32)

    def test_shift_right(self):
        # Insert 9 at index 0
        expected = torch.tensor([[9, 1, 2], 
                                 [9, 4, 5]])
        result = shc_loss.shift_tensor_horizontally(self.x, 9.0, direction='right')
        self.assertTrue(torch.equal(result, expected))

    def test_shift_left(self):
        # Insert 9 at the last index
        expected = torch.tensor([[2, 3, 9], 
                                 [5, 6, 9]])
        result = shc_loss.shift_tensor_horizontally(self.x, 9.0, direction='left')
        self.assertTrue(torch.equal(result, expected))

#class PostProcessingTest(unittest.TestCase):
#
#    @classmethod
#    def setUpClass(cls) -> None:
#        cls._ = {}
#        # B = 2, T = 5, C = 3
#        # yapf: disable
#        cls._ground_truth_prob = {}
#        cls._ground_truth_prob["SEQ_DATA"] = torch.tensor(
#                   [[[0.89, 0.11, 0.00, 0.00],
#                     [0.45, 0.45, 0.05, 0.05],
#                     [0.00, 0.85, 0.15, 0.00],
#                     [0.05, 0.45, 0.45, 0.05],
#                     [0.02, 0.01, 0.96, 0.01]],
#                    [[0.70, 0.20, 0.10, 0.00],
#                     [0.91, 0.03, 0.03, 0.03],
#                     [0.02, 0.94, 0.02, 0.02],
#                     [0.01, 0.01, 0.97, 0.01],
#                     [0.00, 0.00, 0.00, 0.00]]], dtype=torch.float32)
#        # yapf: enable
#        cls._ground_truth_prob["SEQ_LEN"] = torch.tensor([5, 4],
#                                                         dtype=torch.int32)
#
#    def test_apply_postprocessing_entropy_uniform_true(self):
#        ENTROPY_TH = 0.3604
#        UNIFORM_FLAG = True
#        actual, _ = shc_loss.apply_postprocessing(
#            self._ground_truth_prob["SEQ_DATA"],
#            self._ground_truth_prob["SEQ_LEN"],
#            shc_loss.ThresholdType.ENTROPY,
#            ENTROPY_TH,
#            UNIFORM_FLAG,
#        )
#        expected = torch.tensor(
#            [[[1.00, 0.00, 0.00, 0.00],
#              [0.25, 0.25, 0.25, 0.25],
#              [0.25, 0.25, 0.25, 0.25],
#              [0.25, 0.25, 0.25, 0.25],
#              [0.00, 0.00, 1.00, 0.00]],
#             [[0.25, 0.25, 0.25, 0.25],
#              [0.25, 0.25, 0.25, 0.25],
#              [0.00, 1.00, 0.00, 0.00],
#              [0.00, 0.00, 1.00, 0.00],
#              [0.00, 0.00, 0.00, 0.00]]],
#            dtype=torch.float32) # yapf: disable
#
#        self.assertTrue(torch.equal(expected, actual))
#
#    def test_apply_postprocessing_entropy_uniform_false(self):
#        ENTROPY_TH = 0.3604
#        UNIFORM_FLAG = False
#        actual, _ = shc_loss.apply_postprocessing(
#            self._ground_truth_prob["SEQ_DATA"],
#            self._ground_truth_prob["SEQ_LEN"],
#            shc_loss.ThresholdType.ENTROPY,
#            ENTROPY_TH,
#            UNIFORM_FLAG,
#        )
#        expected = torch.tensor(
#            [[[1.00, 0.00, 0.00, 0.00],
#              [0.45, 0.45, 0.05, 0.05],
#              [0.00, 0.85, 0.15, 0.00],
#              [0.05, 0.45, 0.45, 0.05],
#              [0.00, 0.00, 1.00, 0.00]],
#             [[0.70, 0.20, 0.10, 0.00],
#              [0.91, 0.03, 0.03, 0.03],
#              [0.00, 1.00, 0.00, 0.00],
#              [0.00, 0.00, 1.00, 0.00],
#              [0.00, 0.00, 0.00, 0.00]]],
#            dtype=torch.float32) # yapf: disable
#
#        self.assertTrue(torch.equal(expected, actual))
#
#    def test_apply_postprocessing_max_prob_uniform_true(self):
#        MAX_PROB_TH = 0.9
#        UNIFORM_FLAG = True
#        actual, _ = shc_loss.apply_postprocessing(
#            self._ground_truth_prob["SEQ_DATA"],
#            self._ground_truth_prob["SEQ_LEN"],
#            shc_loss.ThresholdType.MAX_PROB,
#            MAX_PROB_TH,
#            UNIFORM_FLAG,
#        )
#        expected = torch.tensor(
#            [[[0.25, 0.25, 0.25, 0.25],
#              [0.25, 0.25, 0.25, 0.25],
#              [0.25, 0.25, 0.25, 0.25],
#              [0.25, 0.25, 0.25, 0.25],
#              [0.00, 0.00, 1.00, 0.00]],
#             [[0.25, 0.25, 0.25, 0.25],
#              [1.00, 0.00, 0.00, 0.00],
#              [0.00, 1.00, 0.00, 0.00],
#              [0.00, 0.00, 1.00, 0.00],
#              [0.00, 0.00, 0.00, 0.00]]],
#            dtype=torch.float32) # yapf: disable
#
#        self.assertTrue(torch.equal(expected, actual))
#
#    def test_apply_postprocessing_max_prob_uniform_false(self):
#        MAX_PROB_TH = 0.9
#        UNIFORM_FLAG = False
#        actual, _ = shc_loss.apply_postprocessing(
#            self._ground_truth_prob["SEQ_DATA"],
#            self._ground_truth_prob["SEQ_LEN"],
#            shc_loss.ThresholdType.MAX_PROB,
#            MAX_PROB_TH,
#            UNIFORM_FLAG,
#        )
#        expected = torch.tensor(
#            [[[0.89, 0.11, 0.00, 0.00],
#              [0.45, 0.45, 0.05, 0.05],
#              [0.00, 0.85, 0.15, 0.00],
#              [0.05, 0.45, 0.45, 0.05],
#              [0.00, 0.00, 1.00, 0.00]],
#             [[0.70, 0.20, 0.10, 0.00],
#              [1.00, 0.00, 0.00, 0.00],
#              [0.00, 1.00, 0.00, 0.00],
#              [0.00, 0.00, 1.00, 0.00],
#              [0.00, 0.00, 0.00, 0.00]]],
#            dtype=torch.float32) # yapf: disable
#
#        self.assertTrue(torch.equal(expected, actual))


def shift_tensor_horizontally(x, fill_value, direction='right'):
    """Shifts tensor elements along the last dimension."""
    res = torch.full_like(x, fill_value)
    if direction == 'right':
        res[:, 1:] = x[:, :-1]
    else:
        res[:, :-1] = x[:, 1:]
    return res

class TestAlphaBetaStability(unittest.TestCase):
    """Verifies numerical correctness of forward-backward computation."""

    def setUp(self):
        """Sets up dimensions and tensors for testing."""
        self.B, self.T, self.L = 3, 6, 5
        self.device = torch.device('cpu')


        # Fixed random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)


        self.log_target_probs = torch.log_softmax(
            torch.randn(self.B, self.T, self.L), dim=-1)
        self.log_delta_probs = torch.log(torch.rand(self.B, self.T))

        self.logit_lens = torch.tensor([6, 5, 4])
        self.target_lens = torch.tensor([5, 5, 3])


    def test_forward_backward_consistency(self):
        """Checks if log_prob(total) is consistent across all time steps."""
        # 1. Run the forward-backward (Assuming your function is defined)
        log_alpha, log_beta, _ = shc_loss.calculate_alpha_beta(
            self.log_target_probs, self.target_lens, self.logit_lens,
            self.log_delta_probs, False)

        for b in range(self.B):
            valid_t = self.logit_lens[b]
            # Across all valid t, logsumexp(alpha + beta) must be constant
            # representing the total log probability of the sequence.
            total_log_target_probs = []
            for t in range(valid_t):
                # Combined posterior log-probability at time t
                prob_t = torch.logsumexp(
                    log_alpha[b, t] + log_beta[b, t], dim=-1)
                total_log_target_probs.append(prob_t.item())
            
            # Variance should be near zero if the algorithm is correct
            variance = np.var(total_log_target_probs)
            self.assertLess(variance, 1e-4, 
                           f"Inconsistent total prob at batch {b}")


    def test_padding_mask(self):
        """Ensures that padded regions remain LOG_0 using tensor operations."""
        log_alpha, _, _ = shc_loss.calculate_alpha_beta(
            self.log_target_probs, self.target_lens, self.logit_lens,
            self.log_delta_probs)

        # Create a mask for padded time steps (B, T)
        t_indices = torch.arange(self.T, device=self.device).expand(self.B, -1)
        padding_mask = t_indices >= self.logit_lens.unsqueeze(1)

        # Check if all values in padded regions are LOG_0
        # This masks log_alpha (B, T, L) with padding_mask (B, T)
        masked_values = log_alpha[padding_mask]
        
        target_log0 = torch.tensor(shc_loss.LOG_0, device=self.device)
        self.assertTrue(
            torch.allclose(masked_values, target_log0, atol=1e-2),
            f"Padded regions in log_alpha are not properly set to {shc_loss.LOG_0}"
        )

if __name__ == '__main__':
    unittest.main()
