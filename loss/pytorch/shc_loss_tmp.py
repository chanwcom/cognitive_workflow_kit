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

EPS = torch.finfo(torch.float32).tiny
#LOG_0 = torch.log(torch.tensor(EPS))

LOG_0 = -706.893623  # float(np.log(1e-307))

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


def calculate_alpha_beta(log_target_probs, target_lens, logit_len, log_delta_prob, normalize=True,
                         trans_table=None):
    """Calculates the alpha and beta variables.

    This calculates the alpha and beta variables required for CTC computation.
    Note that the definition of beta variable is somewhat different from the
    original CTC paper. This equation will be explained in my future paper.
    TODO(chanwcom) Adds the paper link.

    Args:
        log_target_probs: A tensor of posterior probabilities of each label.
            The shape is (batch_size, max_logit_len, max_target_len).
            Mathematically, it is given by the following equation:
                log (p_{[m]}(y_l | x)).
        target_lens: A tensor containing the label lengths.
            The shape is (batch_size).
        logit_len: A tensor containing the logit lengths.
            The shape is (batch_size).
        log_delta_proab: A tensor containing the log transition probability.
            The shape is (batch_size, max_logit_len).
    """
    batch_size = log_target_probs.shape[0]
    max_target_len = torch.max(target_lens)
    max_logit_len = torch.max(logit_len)

    # Initalization of log_alpha and log_beta
    log_alpha = torch.full((batch_size, max_logit_len, max_target_len),
                           fill_value=LOG_0)
    log_beta = torch.full((batch_size, max_logit_len, max_target_len),
                          fill_value=LOG_0)

    # Mask is used for calculating log_beta for proper backward initialization.
    mask = seq_loss_util.sequence_mask(logit_len, maxlen=max_logit_len)

    initial_log_alpha = ((1.0 - (torch.nn.functional.one_hot(
        torch.zeros(size=(batch_size, ), dtype=torch.int64), max_target_len))) *
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

        log_target_probs_shifted = shift_tensor_horizontally(
            log_target_probs[:, t, :], LOG_0, 'right')

#        log_alpha[:, t, :] = torch.log(
#            (torch.exp(prev_log_alpha + log_target_probs[:, t, :])
#            + torch.exp(
#                prev_log_alpha_shifted + log_delta_prob[:, t].unsqueeze(1)))
#                    .clamp(min=EPS))

#        log_alpha[:, t, :] = torch.log(
#            (torch.exp(prev_log_alpha + log_target_probs[:, t, :])
#            + torch.exp(
#                    prev_log_alpha_shifted + log_target_probs[:, t, :])))

#        log_alpha[:, t, :] = torch.log(
#            (torch.exp(prev_log_alpha_shifted + log_target_probs[:, t, :])
#            + torch.exp(
#                prev_log_alpha + log_delta_prob[:, t].unsqueeze(1)))
#                    .clamp(min=EPS))

        transitions = prev_log_alpha.unsqueeze(2) + trans_table
        log_alpha[:, t, :] = torch.logsumexp(transitions, dim=1) + log_target_probs[:, t, :]


        # Normalizes the log sequence prob.
        if normalize: 
            log_alpha_max = torch.max(log_alpha[:, t, :], axis=1,
                                      keepdims=True).values
            log_alpha[:, t, :] -= log_alpha_max
        else:
            log_alpha_max = torch.zeros(batch_size, 1)

        # Accumulates the maximum.
        accum_log_alpha_max[:, t] = (prev_log_alpha_max +
                                     torch.squeeze(log_alpha_max, axis=-1))
        prev_log_alpha_max = accum_log_alpha_max[:, t]

    initial_log_beta = (
        (1.0 - torch.nn.functional.one_hot(target_lens - 1, max_target_len)) *
        LOG_0)
    log_beta[:, max_logit_len - 1, :] = initial_log_beta 

    time_mask = torch.unsqueeze(
        seq_loss_util.sequence_mask(
            logit_len, maxlen=max_logit_len), axis=2) # yapf: disable

    for t in range(max_logit_len - 1, -1, -1):
        tt = min(t + 1, max_logit_len - 1) 
        prev_log_beta = log_beta[:, tt, :]

        # Calculates log_beta recursively from the next time step.
#        prev_log_beta_shifted = shift_tensor_horizontally(
#            prev_log_beta, LOG_0, 'left')

#        prev_log_target_prob_shifted = shift_tensor_horizontally(
#            log_target_probs[:, t+1, :], LOG_0, 'left')

        # Calculates log_beta recursively from the next time step.
#        log_beta[:, t, :] = torch.log(
#            (torch.exp(prev_log_beta + log_target_probs[:, t + 1, :])
#            + torch.exp(prev_log_beta_shifted + log_delta_prob[:, t + 1].unsqueeze(1))).clamp(min=EPS))

#        log_beta[:, t, :] = torch.log(
#            (torch.exp(prev_log_beta + log_target_probs[:, t + 1, :])
#                + torch.exp(prev_log_beta_shifted + prev_log_target_prob_shifted)))

#        log_beta[:, t, :] = torch.log(
#            (torch.exp(prev_log_beta_shifted + prev_log_target_prob_shifted)
#            + torch.exp(prev_log_beta + log_delta_prob[:, t + 1].unsqueeze(1))).clamp(min=EPS))


        messages = (prev_log_beta + log_target_probs[:, tt, :]).unsqueeze(1)
        log_beta[:, t, :] = torch.logsumexp(messages + trans_table, dim=2)
        # Selectively re-initializes log_beta for sequences ending at time 't'.
        #
        # Since samples in a batch have different lengths, we reset log_beta 
        # to the initial state at each sample's respective terminal time step.
        # If mask is zero, then makes the current log_beta zero by multiplying 
        # with the mask. After that, re-initializes the log_beta to be 
        # "initial_log_beta".
        log_beta[:, t, :] = torch.where(
            time_mask[:, tt, :], 
            log_beta[:, t, :],    # True: Maintaining the current value.
            initial_log_beta      # False: Initial value.
        )

        if normalize:
            # Normalize log_beta by subtracting the max value to prevent overflow.
            log_beta_max = torch.max(log_beta[:, t, :], dim=1, keepdim=True).values
            log_beta[:, t, :] -= log_beta_max

    # 3. Post-processing and Final Sequence Probability
    # Apply time mask and label mask for clean output.
    log_alpha.masked_fill_(~time_mask, LOG_0)
    log_beta.masked_fill_(~time_mask, LOG_0)

    label_mask = seq_loss_util.sequence_mask(
        target_lens, max_target_len).unsqueeze(1).to(torch.bool)
    log_alpha.masked_fill_(~label_mask, LOG_0)
    log_beta.masked_fill_(~label_mask, LOG_0)

    log_seq_prob_final = seq_loss_util._calculate_unnormalized_log_seq_prob(
        log_alpha, accum_log_alpha_max, logit_len, target_lens)

    return log_alpha, log_beta, log_seq_prob_final


class ShcLoss(torch.autograd.Function):
    """A class for calculating the SHC loss."""

    @staticmethod
    def forward(ctx,
                labels,
                target_lens,
                logits,
                logits_len, transition_token_id=0):
        """Calculates the Sequential Hypothesis Classifier (SHC) loss.

        Args:
            ctx: Contexts for this CtcLoss operation.
            labels: A tensor containing batch of ground-truth label sequences.
                Note that this label sequence should already include blank labels.
                The shape is given by (batch_size, max_target_len).
            target_lens: The lengths of labels that has the shape of
                (batch_size).
            logits: The predicted "logit value". The shape is given by
                (batch_size, max_logit_seq_len, num_classes).
            logits_len: The len of logits that has the shape of (batch_size).

        Note that zero values are assumed to be masked-values.

        Returns:
            A tuple containing (loss, grad)
        """
        # Checks whether the shape of labels is (B, L).
        assert labels.dim() == 2

        # Checks whether the shape of logits is (B, T, C)
        assert logits.dim() == 3

        # Checks the consistency of the batch size.
        assert labels.shape[0] == logits.shape[0]

        # Start of TODO TODO
        inputs = {}
        inputs["SEQ_DATA"] = labels
        inputs["SEQ_LEN"] = target_lens

        inputs = seq_loss_util.to_blank_augmented_labels(inputs, 0, True, False)

        labels = inputs["SEQ_DATA"]
        target_lens = inputs["SEQ_LEN"]
        # End of TODO TODO

        clamped_labels = torch.clamp(labels, min=0)

        # Converting the sequences.
        # Note that the following is only for HuggingFace case.
        # In case of HuggingFace, the boundary blanks should be added and non
        # -blank token indices should NOT be updated.
        log_target_probs = seq_loss_util.calculate_log_label_prob(
            clamped_labels, torch.softmax(logits, dim=-1))

        

        trans_table = seq_loss_util.label_trans_allowance_table(
            labels, target_lens, seq_loss_util.LabelType.CTC)

        # Alpha and beta should be calculated.
        log_delta_prob = torch.log(torch.softmax(logits, dim=-1))[:, :, transition_token_id]
        log_alpha, log_beta, log_seq_prob = calculate_alpha_beta(
            log_target_probs, target_lens, logits_len, log_delta_prob, True, trans_table)

        # "gamma" is the posterior probability of the alignment variable $q_t$.
        #
        # The "alignment variable" $q_t$ is a random variable representing
        # the distribution  of the label sequcne index $l$ at time $t$.
        #
        # gamma is defined by:
        #   p(\mathbf{q_t} = l | \mathbbm{x}, \mathbbm{y}).
        #
        # gamma can be expressed in terms of \alpha and \beta as follows:
        #   gamma_{t, l} = sum_{l \in {l | q_t = l}} \alpha_{t, l} \beta{t, l}
        #                / sum_{l=0^L-1} \alpha_{t, l} \beta{t, l}.
        #
        # log_gamma is defined as follows:
        #   log p(q_t = l| x, y) where t is the temporal index, and l is the
        # blank-augmented label sequence index.
        # The shape of log_gamma is (batch_size, max_logits_len, max_target_len).
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - torch.logsumexp(log_gamma, axis=2, keepdim=True)

        # To ignore an invalid loss case.
        #
        # If target_lens < logits_len, then the loss is not valid.
        invalid_length_mask = (torch.greater_equal(
            logits_len, target_lens)).type(torch.float32)

        loss = -torch.multiply(log_seq_prob, invalid_length_mask)

        max_target_len = torch.max(target_lens)
        num_classes = logits.shape[2]
        log_ground_truth_prob = torch.ones_like(logits,
                                                dtype=torch.float32) * LOG_0

        # Calculates an estimated time-aligned ground-truth sequence.
        #
        # log_ground_truth_prob is \tilde{\mathbbm{y}_t}.
        #
        # Update is done for each label to reduce memory requirement.
        # TODO(chanwcom)Is it really true?
        # Check with real codes.
        for l in range(max_target_len):
            onehot = (1.0 - (torch.nn.functional.one_hot(
                clamped_labels[:, l], num_classes))) * LOG_0

            # For specific "l", multiply gamma_{t, l} with one_hot(c_l).
            #
            # For each example in a batch, it becomes a vector where the
            # c_l element has the value of gamma{t, l}.
            # Note that c_l is "j", which is the class index.
            # Since logarithm is used, multiplictaion is changed with addition.
            updates = (torch.unsqueeze(log_gamma[:, :, l], axis=2) +
                       torch.unsqueeze(onehot, axis=1))
            log_ground_truth_prob = torch.logaddexp(log_ground_truth_prob,
                                                    updates)

        ground_truth_prob = torch.exp(log_ground_truth_prob)

        gradient = -(ground_truth_prob - torch.softmax(logits, dim=2))

        # To ignore an invalid loss case.
        #
        # If target_lens < logits_len, then the loss is not valid.
        invalid_length_mask = (torch.greater_equal(
            logits_len, target_lens)).type(torch.float32)
        gradient = torch.multiply(
            gradient, torch.reshape(invalid_length_mask, (-1, 1, 1)))

        # Seqeunce mask
        seq_mask = seq_loss_util.sequence_mask(logits_len,
                                 maxlen=torch.max(logits_len))

        # The dimension of "gradient" is (batch_size, logit_len, num_classes)
        gradient = torch.multiply(gradient, torch.unsqueeze(seq_mask, axis=2))

        ctx.save_for_backward(gradient)

        return loss

    @staticmethod
    def backward(ctx, grad):
        gradient, = ctx.saved_tensors
        gradient = torch.multiply(gradient, torch.reshape(grad, (-1, 1, 1)))

        return None, None, gradient, None
