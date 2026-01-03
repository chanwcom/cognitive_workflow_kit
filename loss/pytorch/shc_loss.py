"""A module implementing utilities for sequence losses."""

# pylint: disable=no-member, invalid-name, import-error

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"
# Standard imports
import enum
from typing import Literal

# Third-party imports
import numpy as np
import torch

# Custom imports
from loss.pytorch import seq_loss_util

# TODO(chanwcom) Replace with this one. But unit tests need to be updated.
#LOG_00 = torch.tensor(np.log(np.finfo(np.float64).tiny).astype(np.float32))

LOG_0 = torch.tensor(np.log(1e-307)).type(torch.float32)


def shift_tensor_horizontally(
    x: torch.Tensor, 
    fill_value: float,
    direction: Literal['left', 'right'] = 'right'
) -> torch.Tensor:
    """Inserts a column and shifts existing ones, maintaining shape (B, L).

    Args:
        x: The input tensor of shape (B, L).
        fill_value: The value for the new column.
        direction: Which direction to insert the new column ('left' or 'right').
            'left': New column at index 0, last column dropped.
            'right': New column at last index, first column dropped.

    Returns:
        A tensor of shape (B, L) with the shifted data.
    """
    batch_size, _ = x.shape
    new_col = torch.full(
        (batch_size, 1), fill_value, dtype=x.dtype, device=x.device
    )

    if direction == 'right':
        # New column at the start, drop the last column
        return torch.cat([new_col, x[:, :-1]], dim=1)
    elif direction == 'left':
        # New column at the end, drop the first column
        return torch.cat([x[:, 1:], new_col], dim=1)
    else:
        raise ValueError("direction must be either 'left' or 'right'")


def calculate_alpha_beta(label_trans_table, log_label_prob, label_len,
                         logit_len, log_delta_prob):
    """Calculates the alpha and beta variables.

    This calculates the alpha and beta variables required for CTC computation.
    Note that the definition of beta variable is somewhat different from the
    original CTC paper. This equation will be explained in my future paper.
    TODO(chanwcom) Adds the paper link.

    Args:
        label_trans_table: A tensor containing the transition tables.
            The shape is (batch_size, max_label_seq_len, max_label_seq_len).
        log_label_prob: A tensor of posterior probabilities of each label.
            The shape is (batch_size, max_logit_len, max_label_len).
            Mathematically, it is given by the following equation:
                log (p_{[m]}(y_l | x)).
        label_len: A tensor containing the label lengths.
            The shape is (batch_size).
        logit_len: A tensor containing the logit lengths.
            The shape is (batch_size).
        log_delta_proab: A tensor containing the log transition probability.
            The shape is (batch_size, max_logit_len).
    """
    batch_size = log_label_prob.shape[0]
    max_label_len = torch.max(label_len)
    max_logit_len = torch.max(logit_len)

    # Initalization of log_alpha and log_beta
    log_alpha = torch.full((batch_size, max_logit_len, max_label_len),
                           fill_value=LOG_0)
    log_beta = torch.full((batch_size, max_logit_len, max_label_len),
                          fill_value=LOG_0)

    # Mask is used for calculating log_beta for proper backward initialization.
    mask = seq_loss_util.sequence_mask(logit_len, maxlen=max_logit_len)

    initial_log_alpha = ((1.0 - (torch.nn.functional.one_hot(
        torch.zeros(size=(batch_size, ), dtype=torch.int64), max_label_len))) *
                      LOG_0)

    accum_log_alpha_max = torch.zeros((batch_size, max_logit_len),
                                      dtype=torch.float32)
    prev_log_alpha_max = torch.zeros((batch_size), dtype=torch.float32)
    log_alpha[:, 0, :] = initial_log_alpha

    for t in range(1, max_logit_len):
        prev_log_alpha = log_alpha[:, t - 1, :]
        # Calculates log_alpha recursively from the previous time step.
        prev_log_alpha_shifted = shift_tensor_horizontally(
            prev_log_alpha, LOG_0, 'right')

        log_alpha[:, t, :] = torch.log(
            torch.exp(prev_log_alpha + log_label_prob[:, t, :])
            + torch.exp(prev_log_alpha_shifted + log_delta_prob[:, t].unsqueeze(1)))

        # Normalizes the log sequence prob.
        log_alpha_max = torch.max(log_alpha[:, t, :], axis=1,
                                  keepdims=True).values
        log_alpha[:, t, :] -= log_alpha_max

        # Accumulates the maximum.
        accum_log_alpha_max[:, t] = (prev_log_alpha_max +
                                     torch.squeeze(log_alpha_max, axis=-1))
        prev_log_alpha_max = accum_log_alpha_max[:, t]

    initial_log_beta = (
        (1.0 - torch.nn.functional.one_hot(label_len - 1, max_label_len)) *
        LOG_0)
    log_beta[:, max_logit_len - 1, :] = initial_log_beta 

    time_mask = torch.unsqueeze(
        seq_loss_util.sequence_mask(
            logit_len, maxlen=max_logit_len),
        axis=2) # yapf: disable

    next_log_label_prob = torch.zeros(size=(batch_size, max_label_len))
    for t in range(max_logit_len - 2, -1, -1):
        prev_log_beta = log_beta[:, t + 1, :]
        # Calculates log_beta recursively from the next time step.
        prev_log_beta_shifted = shift_tensor_horizontally(
            prev_log_beta, LOG_0, 'right')

        # Calculates log_beta recursively from the next time step.
        log_beta[:, t, :] = torch.log(
            torch.exp(prev_log_beta + log_label_prob[:, t + 1, :])
            + torch.exp(prev_log_beta_shifted + log_delta_prob[:, t + 1].unsqueeze(1)))


        # Normalize log_beta by subtracting the max value to prevent overflow.
        log_beta_max = torch.max(log_beta[:, t, :], dim=1, keepdim=True).values
        log_beta[:, t, :] -= log_beta_max

        # Selectively re-initializes log_beta for sequences ending at time 't'.
        #
        # Since samples in a batch have different lengths, we reset log_beta 
        # to the initial state at each sample's respective terminal time step.
        # If mask is zero, then makes the current log_beta zero by multiplying 
        # with the mask. After that, re-initializes the log_beta to be 
        # "initial_log_beta".
        log_beta[:, t, :] = torch.where(
            time_mask[:, t, :], 
            log_beta[:, t, :],    # True: Maintaining the current value.
            initial_log_beta      # False: Initial value.
        )

    # 3. Post-processing and Final Sequence Probability
    # Apply time mask and label mask for clean output.
    log_alpha.masked_fill_(~time_mask, LOG_0)
    log_beta.masked_fill_(~time_mask, LOG_0)

    label_mask = seq_loss_util.sequence_mask(
        label_len, max_label_len).unsqueeze(1).to(torch.bool)
    log_alpha.masked_fill_(~label_mask, LOG_0)
    log_beta.masked_fill_(~label_mask, LOG_0)

    # We utilize the "tf.stop_gradient" API with the "tf.nest.map_structure"
    # API based on the recommendation in the following page:
    # https://www.tensorflow.org/api_docs/python/tf/scan

    log_seq_prob_final = seq_loss_util._calculate_unnormalized_log_seq_prob(
        log_alpha, accum_log_alpha_max, logit_len, label_len)

    return log_alpha, log_beta, log_seq_prob_final


