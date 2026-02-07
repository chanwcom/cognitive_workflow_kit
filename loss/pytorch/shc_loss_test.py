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




class TestTransTable(unittest.TestCase):
    """Unit tests for create_trans_table with sub_label_factor."""

    def setUp(self):
        self.batch_size = 2
        self.max_len = 12
        self.labels = torch.zeros((self.batch_size, self.max_len))
        self.labels_len = torch.tensor([self.max_len, self.max_len])

    def test_factor_2(self):
        """Tests if factor 2 allows skipping odd-indexed elements."""
        factor = 2
        table = shc_loss.create_trans_table(self.labels_len, factor)[0]
        # Skipped element (i+1) should be 1, 3, 5... (2n-1)
        # So i should be 0, 2, 4...
        self.assertEqual(table[0, 2], 0)  # 1 skipped
        self.assertEqual(table[2, 4], 0)  # 3 skipped
        self.assertEqual(table[1, 3], shc_loss.LOG_0)  # 2 is not (2n-1)

    def test_factor_4(self):
        """Tests if factor 4 allows skipping (4n-1) indexed elements."""
        factor = 4
        table = shc_loss.create_trans_table(self.labels_len, factor)[0]
        # Skipped element (i+1) should be 3, 7, 11... (4n-1)
        # So i should be 2, 6, 10...
        self.assertEqual(table[2, 4], 0)  # 3 skipped
        self.assertEqual(table[6, 8], 0)  # 7 skipped
        self.assertEqual(table[0, 2], shc_loss.LOG_0)  # 1 is not (4n-1)
        self.assertEqual(table[4, 6], shc_loss.LOG_0)  # 5 is not (4n-1)

    def test_self_loops(self):
        """Tests if self-loops are allowed."""
        table = shc_loss.create_trans_table(self.labels_len, 2)[0]
        self.assertEqual(table[0, 0], 0)
        self.assertEqual(table[1, 1], 0)
        self.assertEqual(table[3, 3], 0)
        self.assertEqual(table[2, 2], 0)


class TestBlockAugmentation(unittest.TestCase):
    def setUp(self):
        # 2 classes (0, 1), n=3 augmentation
        self.num_classes = 10
        self.inputs = {
            "SEQ_DATA": torch.tensor([[1, 2]], dtype=torch.long),
            "SEQ_LEN": torch.tensor([2], dtype=torch.long)
        }

    def test_block_logic(self):
        n = 3
        k = self.num_classes
        result = shc_loss.to_onset_block_augmented_n(self.inputs, n, k)
        
        # Expected: [1, 2, 1+k, 2+k, 1+2k, 2+2k] -> [1, 2, 11, 12, 21, 22]
        expected_data = torch.tensor([[1, 11, 21, 2, 12, 22]], dtype=torch.long)
        torch.testing.assert_close(result["SEQ_DATA"], expected_data)

    def test_masking(self):
        # Test if padding is preserved across blocks
        inputs_with_pad = {
            "SEQ_DATA": torch.tensor([[5, 0]], dtype=torch.long),
            "SEQ_LEN": torch.tensor([1], dtype=torch.long)
        }
        n = 2
        k = self.num_classes
        result = shc_loss.to_onset_block_augmented_n(inputs_with_pad, n, k)

        # Expected: [5, 0, 15, 0] -> Only elements within SEQ_LEN are augmented
        # After masking: [5, 0, 15, 0] (Wait, block-wise padding is tricky)
        # The logic ensures only valid time-steps are considered.
        expected_data = torch.tensor([[5, 15, 0, 0]], dtype=torch.long)
        torch.testing.assert_close(result["SEQ_DATA"], expected_data)


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
        self.B, self.T, self.L = 3, 10, 9
        self.device = torch.device('cpu')

        # Fixed random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        self.log_target_probs = torch.log_softmax(
            torch.randn(self.B, self.T, self.L), dim=-1)

        self.logit_lens = torch.tensor([10, 8, 6])
        self.target_lens = torch.tensor([9, 6, 3])
        self.sub_label_factor = 3
        self.trans_table = shc_loss.create_trans_table(
            self.target_lens, self.sub_label_factor)

    def test_forward_backward_consistency(self):
        """Checks if log_prob(total) is consistent across all time steps."""

        # 1. Run the forward-backward (Assuming your function is defined)
        log_alpha, log_beta, _ = shc_loss.calculate_alpha_beta(
            self.log_target_probs, self.target_lens, self.logit_lens, False,
            self.trans_table)

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
            False, self.trans_table)

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
