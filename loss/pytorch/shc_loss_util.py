"""A module implementing utilities for sequence losses."""

# pylint: disable=no-member, invalid-name, import-error


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"
# Standard imports

import enum

# Third-party imports
import numpy as np
import torch

from cwk.loss.pytorch import sals_smoothing

# TODO(chanwcom) Replace with this one. But unit tests need to be updated.
#LOG_00 = torch.tensor(np.log(np.finfo(np.float64).tiny).astype(np.float32))

LOG_0 = -706.893623  # float(np.log(1e-307))
#LOG_0 = -100.00  


#@torch.jit.script
def apply_post_processing(ground_truth_prob, logits_len, alpha, beta):
    """Applies a specialized label smoothing to the ground truth probabilities.

    The smoothing formula is:
        P' = (1 - alpha - beta) * P + alpha * u1 + beta * u2
    where:
        u1: Uniform distribution over non-masked classes at time t.
        u2: Uniform distribution over masked classes at time t.

    Args:
        ground_truth_prob (torch.Tensor): GT probabilities of shape (B, T, C).
        logits_len (torch.Tensor): Actual lengths of each sequence (B).
        alpha (float): Smoothing coefficient for non-masked classes.
        beta (float): Smoothing coefficient for masked classes.

    Returns:
        torch.Tensor: The smoothed probability tensor.
    """
    device = ground_truth_prob.device
    dtype = ground_truth_prob.dtype
    batch_size, max_t, num_classes = ground_truth_prob.shape

    # 1. Define masks for valid (non-LOG_0) and masked entries.
    # We assume ground_truth_prob is in probability domain [0, 1].
    is_valid = (ground_truth_prob > 0.0).to(dtype)
    is_masked = (1.0 - is_valid)

    # 2. Calculate counts for uniform distribution denominators.
    eps = 1e-8
    n_valid = torch.sum(is_valid, dim=-1, keepdim=True)
    n_masked = torch.sum(is_masked, dim=-1, keepdim=True)

    # 3. Construct uniform distributions u1 and u2.
    # u1: 1/n_valid for valid indices, 0 otherwise.
    u1 = is_valid / (n_valid + eps)
    # u2: 1/n_masked for masked indices, 0 otherwise.
    u2 = is_masked / (n_masked + eps)

    # 4. Apply the specialized smoothing formula.
    smoothed_prob = (
        (1.0 - alpha - beta) * ground_truth_prob + 
        alpha * u1 + 
        beta * u2
    )

    # 5. Zero out padded time steps using a sequence mask.
    range_t = torch.arange(max_t, device=device).unsqueeze(0)
    time_mask = (range_t < logits_len.unsqueeze(1)).to(dtype).unsqueeze(2)
    smoothed_prob = smoothed_prob * time_mask

    return smoothed_prob
