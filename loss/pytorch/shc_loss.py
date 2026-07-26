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
from cwk.loss.pytorch import seq_loss_util

# TODO(chanwcom) Replace with this one. But unit tests need to be updated.
#LOG_00 = torch.tensor(np.log(np.finfo(np.float64).tiny).astype(np.float32))

#EPS = torch.finfo(torch.float32).tiny
EPS = 0
#LOG_0 = torch.log(torch.tensor(EPS))

# Approximate log(0) to prevent underflow 
#
# This value corresponds to the lower limit of float precision (~1e-307).
#
# The other options may be follows:
# LOG_0 = torch.log(torch.tensor(EPS))
# LOG_0 = torch.tensor(np.log(np.finfo(np.float64).tiny).astype(np.float32))
#
# However, it is better to use a fixed value rather than a value depending on system or configurations.
LOG_0 = -706.893623  # float(np.log(1e-307))

def enforce_log_zero(
    tensor: torch.Tensor,
    log_zero: float = LOG_0,
    tol: float = 1e-5,
) -> torch.Tensor:
    """Enforces exact numerical flooring for log-space zero values.

    During log-domain arithmetic (e.g., log-space matrix multiplications,
    additions, or recurrent forward-backward updates), floating-point rounding
    errors can cause values that represent exact zeros (represented by
    `LOG_0`) to drift slightly (e.g., `LOG_0 - 1e-6` or `LOG_0 + 1e-6`). 
    This function detects values near or below `LOG_0` within a specified 
    tolerance and forces them to the exact `LOG_0` constant to preserve 
    numerical stability and ensure accurate conditional masking.

    Use Cases:
        - Post-processing sequence modeling variables (such as CTC alpha/beta 
          trellis states) to prevent precision drift from accumulating across 
          time-step iterations.
        - Cleaning up tensors prior to logical comparisons or downstream 
          log-sum-exp reductions where exact boundary matching is required.

    Args:
        tensor (torch.Tensor): The input tensor containing log-probabilities 
            or log-domain accumulation values.
        log_zero (float, optional): The constant value representing log(0). 
            Defaults to `LOG_0`.
        tol (float, optional): The tolerance threshold to account for 
            floating-point drift around `log_zero`. Defaults to `1e-5`.

    Returns:
        torch.Tensor: The modified tensor with drifted log-zero values 
            strictly floored and replaced by the exact `log_zero` constant.
    """
    is_log_zero = tensor <= (log_zero + tol)
    return torch.where(is_log_zero, log_zero, tensor)

def create_trans_allowance_table_shc(
    token_seq, boundary_token_id, blank_token_id, log_0=-1e10
):
    """Constructs a transition allowance table for SHC in parallel.

    Args:
        token_seq: Tensor of shape (batch_size, seq_len).
        boundary_token_id: ID for boundary tokens (stays only for 1 frame).
        blank_token_id: ID for blank tokens (allows i -> i+2 skip).
        log_0: Value to represent disallowed transitions.

    Returns:
        A tensor of shape (batch_size, seq_len, seq_len) where 0.0 indicates
        an allowed transition and log_0 indicates a disallowed one.
    """
    batch_size, max_seq_len = token_seq.shape
    device = token_seq.device

    # Initialize the table with log_0 (all transitions blocked by default).
    trans_table = torch.full(
        (batch_size, max_seq_len, max_seq_len),
        fill_value=float(log_0),
        device=device,
    )

    # Prepare indices for vectorization.
    idx = torch.arange(max_seq_len, device=device)
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)

    # Rule 1: i -> i (Self-loop).
    # Allowed only if the current token is NOT a boundary_token_id.
    self_loop_mask = (token_seq != boundary_token_id)
    # Use advanced indexing to update diagonal elements across the batch.
    trans_table[batch_idx, idx, idx] = torch.where(
        self_loop_mask, 0.0, float(log_0)
    )

    # Rule 2: i -> i + 1 (Next-step).
    # Always allowed for all tokens except the last index (broadcasted).
    if max_seq_len > 1:
        i_indices = idx[:-1]
        j_indices = idx[1:]
        trans_table[:, i_indices, j_indices] = 0.0

    # Rule 3: i -> i + 2 (Skip-step).
    # Allowed only if the intermediate token (i + 1) is a blank_token_id.
    if max_seq_len > 2:
        # mid_tokens corresponds to index i + 1.
        mid_tokens = token_seq[:, 1:-1]
        skip_mask = (mid_tokens == blank_token_id)
        
        # Get batch and sequence coordinates where skip_mask is True.
        b_indices, s_indices = torch.where(skip_mask)
        
        # Update trans_table for i -> i + 2 transitions using found indices.
        trans_table[b_indices, s_indices, s_indices + 2] = 0.0

    return trans_table

def to_shc_token_seq(
    inputs: dict,
    boundary_token_id: int,
    blank_token_id: int,
    special_token_ids: list
) -> dict:
    """Inserts boundary and blank tokens around normal tokens.

    For each token T, if T is not in special_token_ids, it is transformed 
    into [boundary_token_id, T, blank_token_id]. Otherwise, T remains as is.

    Args:
        inputs: A dict containing "SEQ_DATA" (B, L) and "SEQ_LEN" (B,).
        boundary_token_id: ID to insert before the real token.
        blank_token_id: ID to insert after the real token.
        special_token_ids: List of IDs (e.g., <s>, </s>) to skip expansion.

    Returns:
        A dict with augmented "SEQ_DATA" and updated "SEQ_LEN".
    """
    data = inputs["SEQ_DATA"]
    lengths = inputs["SEQ_LEN"]
    batch_size = data.size(0)
    special_set = set(special_token_ids)
    
    new_sequences = []
    new_lengths = []


    for i in range(batch_size):
        original_seq = data[i, :lengths[i]].tolist()
        augmented_seq = []
        

        for token in original_seq:
            if token in special_set:
                augmented_seq.append(token)
            else:
                augmented_seq.extend([
                    boundary_token_id, 
                    token, 
                    blank_token_id
                ])
        
        new_sequences.append(torch.tensor(augmented_seq))
        new_lengths.append(len(augmented_seq))

    # Pad sequences to the same length
    max_new_len = max(new_lengths)
    padded_data = torch.full(
        (batch_size, max_new_len), 
        fill_value=-100,
        dtype=data.dtype, 
        device=data.device
    )

    for i, seq in enumerate(new_sequences):
        padded_data[i, :new_lengths[i]] = seq

    return {
        "SEQ_DATA": padded_data,
        "SEQ_LEN": torch.tensor(new_lengths, device=lengths.device)
    }

# --- Unit Test ---
def to_onset_block_augmented_n(
    inputs: dict, sub_label_factor: int, num_classes: int
) -> dict:
    """Augments labels with offsets and inserts zeros at specific intervals.

    The labels are augmented to: [L+0*K, L+1*K, ..., 0] where K is num_classes.
    The zero is placed at the index (n * sub_label_factor - 1).

    Args:
        inputs: A dict containing "SEQ_DATA" (batch, seq_len) and "SEQ_LEN".
        sub_label_factor: Expansion factor (e.g., 2 or 3).
        num_classes: The original number of classes (offset step K).

    Returns:
        A dict with augmented "SEQ_DATA" and updated "SEQ_LEN".
    """
    assert isinstance(inputs, dict)
    data = inputs["SEQ_DATA"]
    batch_size, seq_len = data.shape

    # 1. Interleave the original data: [A, B] -> [A, A, A, B, B, B]
    output_data = data.repeat_interleave(sub_label_factor, dim=1)

    # 2. Prepare offsets: [0, K, 2K, ..., (F-1)K] -> repeat for seq_len
    # Example for factor 3: [0, K, 2K, 0, K, 2K, ...]
    single_block_offsets = torch.arange(
        sub_label_factor, device=data.device
    ) * num_classes
    offsets = single_block_offsets.repeat(seq_len)
    
    # 3. Apply offsets to the interleaved data
    output_data = output_data + offsets

    # 4. Zero out the last element of each sub-label block
    # Indices where (i + 1) % sub_label_factor == 0 should be 0.
    mask_indices = torch.arange(
        sub_label_factor - 1, output_data.shape[1], sub_label_factor
    )
    output_data[:, mask_indices] = 0

    # 5. Update lengths and mask padding
    new_lens = inputs["SEQ_LEN"] * sub_label_factor
    max_len = output_data.shape[1]
    range_tensor = torch.arange(max_len, device=data.device)[None, :]
    length_mask = range_tensor < new_lens[:, None]
    output_data = output_data * length_mask.long()

    return {
        "SEQ_DATA": output_data,
        "SEQ_LEN": new_lens
    }

def create_trans_table(labels_len, sub_label_factor):
    """Constructs a table containing the label transition allowance flags.

    Args:
        labels_len: A tensor containing the length of each label sequence.
            The shape is (batch_size).
        sub_label_factor: An integer factor (>= 2). Skip transition (i -> i+2)
            is allowed only if the skipped index (i+1) is a multiple of
            sub_label_factor minus 1 (i.e., i+1 = n * sub_label_factor - 1).

    Returns:
        A tensor containing flags whether transitions are allowed.
            The shape is (batch_size, max_seq_len, max_seq_len).
            a[b, i, j] = 0 if allowed, else LOG_0.
    """
    max_seq_len = torch.max(labels_len).item()
    batch_size = labels_len.shape[0]

    trans_table = torch.full((max_seq_len, max_seq_len), LOG_0)

    # 1. i -> i (Self-loop): Always allowed for all indices.
    indices_i = torch.arange(max_seq_len)
    trans_table[indices_i, indices_i] = 0

    # 2. i -> i + 1 (Step transition): Always allowed within bounds.
    step_idx = torch.arange(max_seq_len - 1)
    trans_table[step_idx, step_idx + 1] = 0

    # 3. i -> i + 2 (Skip transition): Allowed only if (i + 1)
    # is a skip point (n * sub_label_factor - 1).
    n_vals = torch.arange(1, (max_seq_len // sub_label_factor) + 1)
    skip_from = n_vals * sub_label_factor - 2

    # Filter indices: must be >= 0 and i + 2 < max_seq_len.
    valid_mask = (skip_from >= 0) & (skip_from < max_seq_len - 2)
    skip_from = skip_from[valid_mask]

    if skip_from.numel() > 0:
        trans_table[skip_from, skip_from + 2] = 0

    return trans_table.unsqueeze(0).expand(batch_size, -1, -1)
def _calculate_unnormalized_log_seq_prob(log_alpha, accum_log_seq_prob_sum,
                                         logit_len, label_len):
    # In alpha calculation, the log probabilty is normalized to prevent
    # over-flowing and under-flowing. This effect is compensated here.
    # log_p_ctc = log
    batch_size = log_alpha.shape[0]
    batch_index = torch.arange(batch_size, dtype=torch.int32)

    final_log_alpha = log_alpha[batch_index, logit_len - 1, label_len - 1]

    # max(alpha_{T-1,L-1}, alpha_{T-1,L})
    #
    # TODO(chanwcom)
    # There is an issue with the following statement.
    # It should be  addition rather than max.
    # alpha_{T-1,L-2}, alpha_{T-1,L-1}
    # log(Exp(log_alpha_{T-1, L-2}) + Exp(log_alpha_{T-1, L-1}))
#    final_log_alpha = torch.max(final_log_alpha0, final_log_alpha1)

    # Finds the accumulated log seq probability at the last time index.
    final_accum = accum_log_seq_prob_sum[batch_index, logit_len - 1]

    return final_log_alpha + final_accum


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

def _validate_alpha_beta_inputs(
    trans_mask, log_target_probs, target_lens, logit_lens
):
    """Validates the input dimensions and shapes for CTC computation.

    Args:
        trans_mask: A 3D tensor of transition masks.
        log_target_probs: A 3D tensor of log probabilities.
        target_lens: A 1D tensor of target lengths.
        logit_lens: A 1D tensor of logit lengths.

    Raises:
        AssertionError: If any of the input tensors have incorrect dimensions
          or mismatched shapes.
    """
    # Verify the number of dimensions for each tensor.
    assert trans_mask.dim() == 3, (
        f"trans_mask must be 3D, got {trans_mask.dim()}D"
    )
    assert log_target_probs.dim() == 3, (
        f"log_target_probs must be 3D, got {log_target_probs.dim()}D"
    )
    assert target_lens.dim() == 1, (
        f"target_lens must be 1D, got {target_lens.dim()}D"
    )
    assert logit_lens.dim() == 1, (
        f"logit_lens must be 1D, got {logit_lens.dim()}D"
    )

    # Extract shape variables to verify consistency.
    batch_size, max_target_len_1, max_target_len_2 = trans_mask.shape
    b2, max_logit_len, max_target_len_3 = log_target_probs.shape

    # Verify batch size consistency across all inputs.
    assert batch_size == b2, (
        f"Batch size mismatch: trans_mask({batch_size}) vs "
        f"log_target_probs({b2})"
    )
    assert batch_size == target_lens.size(0), (
        f"Batch size mismatch with target_lens: {batch_size} vs "
        f"{target_lens.size(0)}"
    )
    assert batch_size == logit_lens.size(0), (
        f"Batch size mismatch with logit_lens: {batch_size} vs "
        f"{logit_lens.size(0)}"
    )

    # Verify target length consistency (trans_mask must be square in target dims).
    assert max_target_len_1 == max_target_len_2, (
        f"trans_mask must be square in target dims, got "
        f"({max_target_len_1}, {max_target_len_2})"
    )
    assert max_target_len_1 == max_target_len_3, (
        f"max_target_len mismatch: trans_mask({max_target_len_1}) vs "
        f"log_target_probs({max_target_len_3})"
    )


def calculate_alpha_beta(trans_mask, log_target_probs, target_lens,
                         logit_lens):
    """Calculates the alpha and beta variables required for CTC computation.

    Note that the definition of beta variable is somewhat different from the
    original CTC paper. This equation will be explained in my future paper.

    TODO(chanwcom): This assumpes that the initial and the final tokens are
        <s> and <\s>. So, it does not allow starting from (0, 0, 1) in case
        of the forward and (0, T-1, L-2) in case of the backward.
    TODO(chanwcom): Adds the paper link.

    Args:
        trans_mask: A tensor containing the CTC transition masks. The shape
          is (batch_size, max_target_len, max_target_len).
        log_target_probs: A tensor of log posterior probabilities of each
          target token. The shape is (batch_size, max_logit_len, max_target_len).

          Mathematically, it is \hat{p}(k_t = c_l \mid X, \theta)
        target_lens: A tensor containing the target lengths (including blanks). 
          The shape is (batch_size).
        logit_lens: A tensor containing the logit lengths (time steps). 
          The shape is (batch_size).

    Returns:
        log_alpha: The calculated forward variable tensor.
        log_beta: The calculated backward variable tensor.
        log_seq_prob_final: The final unnormalized sequence log probabilities.
    """
    batch_size = log_target_probs.shape[0]
    device = log_target_probs.device
    dtype = log_target_probs.dtype
    max_target_len = torch.max(target_lens)
    max_logit_len = torch.max(logit_lens)

    if __debug__:
        # Perform sanity checks on input dimensions and shapes.
        _validate_alpha_beta_inputs(
            trans_mask, log_target_probs, target_lens, logit_lens
        )

    # Initialize log_alpha and log_beta arrays.
    log_alpha = torch.full((batch_size, max_logit_len, max_target_len),
                           fill_value=LOG_0,
                           device=device,
                           dtype=dtype)
    log_beta = torch.full((batch_size, max_logit_len, max_target_len),
                          fill_value=LOG_0,
                          device=device,
                          dtype=dtype)

    # This is the case for starting with <s>.
    # TODO(chanwcom) This assumes that the sentence starts with <s>.
    # It assumes that <s> cannot be bypassed. 
    log_alpha[:, 0, 0] = log_target_probs[:, 0, 0] 

    # Initialize beta variable using target length info.
    batch_indices = torch.arange(batch_size, device=device, dtype=torch.long)
    target_indices = (target_lens - 1).to(torch.long)

    # This is the case for einding with <\s>.
    # TODO(chanwcom) This assumes that the sentence ends with <\s>.
    # It assumes that <\s> cannot be bypassed. 
    log_beta[batch_indices, - 1, target_indices] = 0
    initial_log_beta = log_beta[:, - 1, :]

    # Prepare time mask for padding frames. Shape: [B, T, 1]
    time_mask = seq_loss_util.sequence_mask(
        logit_lens, int(max_logit_len)).unsqueeze(2).to(dtype)

    max_logit_len_int = int(max_logit_len)
    for t_f in range(1, max_logit_len_int):
        # Forward pass.
        # Trans table broadcast: [B, L, 1] + [B, L, L] -> [B, L, L]
        log_alpha[:, t_f, :] = (
            torch.logsumexp(
            log_alpha[:, t_f - 1, :].unsqueeze(2) + trans_mask, dim=1) +
            log_target_probs[:, t_f, :])

        # Backward Pass: Calculates log_beta recursively.
        # t_b is the time index from the last time step to the first one.
        t_b = max_logit_len_int - t_f - 1
        log_beta[:, t_b, :] = torch.logsumexp(
            (log_beta[:, t_b + 1, :] + log_target_probs[:, t_b + 1, :]).unsqueeze(1) +
            trans_mask, dim=2)

        current_mask = time_mask[:, t_b, :]
        log_beta[:, t_b, :] = ((log_beta[:, t_b, :] * current_mask) +
                             (initial_log_beta * (1.0 - current_mask)))

    # Final Sequence Masking: Vectorized masking outside the loop.
    label_mask = seq_loss_util.sequence_mask(
        target_lens, int(max_target_len)).unsqueeze(1).to(dtype)

    final_valid_mask = time_mask * label_mask  # Shape: [B, T, L]
    log_alpha = torch.where(final_valid_mask == 1.0, log_alpha, LOG_0)
    log_beta = torch.where(final_valid_mask == 1.0, log_beta, LOG_0)

    log_seq_prob_final = log_alpha[
        batch_indices, logit_lens -1,  target_lens - 1]

    return log_alpha, log_beta, log_seq_prob_final


class ShcLoss(torch.autograd.Function):
    """A class for calculating the SHC loss."""

    @staticmethod
    def forward(ctx,
                labels,
                target_lens,
                logits,
                logits_len,
                vocab_size=None):
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

        assert vocab_size

        device = logits.device
        dtype = logits.dtype

        inputs = {}
        inputs["SEQ_DATA"] = labels
        inputs["SEQ_LEN"] = target_lens
        org_target_lens = target_lens

#        inputs = to_onset_block_augmented_n(inputs, sub_label_factor, vocab_size)
        inputs = seq_loss_util.to_blank_augmented_labels(inputs, 0, False, False)
        # inputs, boundary, blank, special

        # TODO(chanwcom) Fix this!! Hard-coding
        #inputs = to_shc_token_seq(inputs, logits.shape[2] - 1, 0, [-100, 1, 2])

        labels = inputs["SEQ_DATA"]
        target_lens = inputs["SEQ_LEN"]

        clamped_labels = torch.clamp(labels, min=0)

        batch_size = labels.shape[0]

        if 0:
            trans_table =  create_trans_allowance_table_shc(
                labels, logits.shape[2] - 1, 0, LOG_0,
            )
        else:
            trans_table = seq_loss_util.label_trans_allowance_table_ctc(labels, target_lens)

#        log_probs = torch.log_softmax(logits, dim=-1)

        # Converting the sequences.
        # Note that the following is only for HuggingFace case.
        # In case of HuggingFace, the boundary blanks should be added and non
        # -blank token indices should NOT be updated.
        log_target_probs = seq_loss_util.calculate_log_label_prob(
            clamped_labels, logits)

        # Alpha and beta should be calculated.
        log_alpha, log_beta, log_seq_prob,  = calculate_alpha_beta(
            trans_table, log_target_probs, target_lens, logits_len)


        log_alpha = torch.clamp(log_alpha, min=LOG_0)
        log_beta = torch.clamp(log_beta, min=LOG_0)

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

        # (IMPORTANT)
        # 2 L - 2 is only for SHC labellilng.
        # But even in that case, 
#        invalid_length_mask = (torch.greater_equal(
#            logits_len, 2 * org_target_lens - 2)).type(torch.float32)
        #invalid_length_mask = validity_flag.type(dtype)

        #loss = -torch.multiply(log_seq_prob, invalid_length_mask)
        loss = -log_seq_prob

        max_target_len = torch.max(target_lens)
        num_classes = logits.shape[2]

        # --- (여기서부터 교체 시작) ---
        # 1. log_gamma를 확률 도메인으로 변환 (B, T, L)
        gamma = torch.exp(log_gamma)

        import pdb; pdb.set_trace()

        # 2. 결과 저장용 텐서 초기화 (B, T, C)
        # C가 작으므로(32~128) 메모리 부담이 거의 없음
        ground_truth_prob = torch.zeros_like(logits)

        # 3. labels를 (B, L) -> (B, T, L)로 확장 (메모리 복사 없는 View 생성)
        # clamped_labels는 (batch_size, max_target_len)
        expanded_indices = clamped_labels.unsqueeze(1).expand(-1, logits.size(1), -1)

        # 4. 핵심 연산: 한 방에 각 클래스 위치에 확률값 더하기
        # dim=2 (클래스 차원)에 대해 인덱스 위치에 gamma 값을 누적
        ground_truth_prob.scatter_add_(2, expanded_indices, gamma)

        # 5. 최종 그라디언트 계산
        gradient = -(ground_truth_prob - torch.softmax(logits, dim=2))
        # --- (여기까지 교체 완료) ---

        # To ignore an invalid loss case.
        #
        # If target_lens < logits_len, then the loss is not valid.
   #     gradient = gradient * invalid_length_mask.view(-1, 1, 1)

        # Seqeunce mask
        seq_mask = seq_loss_util.sequence_mask(logits_len,
                                 maxlen=int(torch.max(logits_len)))

        # The dimension of "gradient" is (batch_size, logit_len, num_classes)
        gradient = torch.multiply(gradient, torch.unsqueeze(seq_mask, axis=2))

        ctx.save_for_backward(gradient)

        return loss

    @staticmethod
    def backward(ctx, grad):
        gradient, = ctx.saved_tensors

        gradient = torch.multiply(gradient, torch.reshape(grad, (-1, 1, 1)))

        return None, None, gradient, None, None
