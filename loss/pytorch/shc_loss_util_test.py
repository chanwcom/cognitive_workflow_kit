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

class TestShcPostProcessing(unittest.TestCase):
    """Unit tests for specialized label smoothing."""

    def setUp(self):
        self.batch_size = 2
        self.max_t = 3
        self.num_classes = 5
        self.logits_len = torch.tensor([2, 3])
        
        # Initialize ground_truth_prob (B, T, C)
        self.gt_prob = torch.zeros(
            (self.batch_size, self.max_t, self.num_classes)
        )

        self.gt_prob = torch.tensor([
            # --- Sample 0 ---
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],  # T=0: Class 1 (1.0)
                [0.0, 0.0, 0.5, 0.5, 0.0],  # T=1: Class 2, 3 (0.5 each)
                [0.0, 0.0, 0.0, 0.0, 0.0]   # T=2: Empty (Padding)
            ],
            # --- Sample 1 ---
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],  # T=0: Class 1 (1.0)
                [0.0, 0.0, 0.5, 0.5, 0.0],  # T=1: Class 2, 3 (0.5 each)
                [1.0, 0.0, 0.0, 0.0, 0.0]   # T=2: Class 0 (1.0)
            ]
        ]) 

    def test_sum_to_one(self):
        """Valid time steps should sum to 1.0."""
        alpha, beta = 0.1, 0.05
        res = shc_loss_util.apply_post_processing(self.gt_prob, self.logits_len, alpha, beta)
        sums = torch.sum(res, dim=-1)
        
        # Sample 0, T=0,1 should be 1.0
        self.assertTrue(torch.allclose(sums[0, :2], torch.ones(2)))
        # Sample 1, T=0,1,2 should be 1.0
        self.assertTrue(torch.allclose(sums[1, :3], torch.ones(3)))
        # Padding should be 0.0
        self.assertEqual(sums[0, 2].item(), 0.0)

    def test_smoothing_math(self):
        """Verify the formula P' = (1-a-b)P + a*u1 + b*u2."""
        alpha, beta = 0.1, 0.2
        res = shc_loss_util.apply_post_processing(self.gt_prob, self.logits_len, alpha, beta)
        
        # t=0, class 1: n_valid=1, n_masked=4
        # P' = (0.7)*1.0 + 0.1*(1/1) + 0.2*(0) = 0.8
        self.assertAlmostEqual(res[0, 0, 1].item(), 0.8)
        
        # t=0, class 0 (masked):
        # P' = (0.7)*0.0 + 0.1*(0) + 0.2*(1/4) = 0.05
        self.assertAlmostEqual(res[0, 0, 0].item(), 0.05)


if __name__ == "__main__":
    unittest.main()
