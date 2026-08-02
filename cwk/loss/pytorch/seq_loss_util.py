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

# TODO(chanwcom) Replace with this one. But unit tests need to be updated.
#LOG_00 = torch.tensor(np.log(np.finfo(np.float64).tiny).astype(np.float32))

LOG_0 = -706.893623  # float(np.log(1e-307))

@torch.jit.script
def sequence_mask(lengths: torch.Tensor, maxlen: int):
    """Applies sequence masking with TorchScript compatibility.
    
    Args:
        lengths: A tensor of shape (batch_size,) containing sequence lengths.
        maxlen: The maximum length of the sequences (must be an int for JIT).
        
    Returns:
        A boolean mask tensor of shape (batch_size, maxlen).
    """
    # Create a row vector: [0, 1, 2, ..., maxlen-1]
    row_vector = torch.arange(0, maxlen, step=1, device=lengths.device)
    
    # Expand lengths to (batch_size, 1) for broadcasting
    matrix = lengths.unsqueeze(1)
    
    # Compare to create the mask
    mask = row_vector < matrix
    
    return mask 


def to_blank_augmented_labels(
        inputs: dict, blank_index: int=0, boundary_blanks: bool=True,
        update_non_blank_token_index: bool=True) -> dict:  # yapf: disable
    """Expands the input sequence with blank labels.

    The blank symbol is inserted at the beginning and at the end of the
    sequences, as well as between labels. If boundary_blanks is False, then
    blank labels are not inserted at the beginning and the end of the sequence.

    Args:
        inputs: A dict containing the input sequence.
            SEQ_DATA: A sparse tensor containing ground truth values.
                The shape is (batch_size, sequence_length).
            SEQ_LEN: A tensor of rank one containing the length of each
                ground truth sequence. The shape is (batch_size).
        blank_index:
            An integer for the blank label in the CTC loss.
        boundary_blanks:
            A boolean flag to insert labels at the boundaries of the sequence.
        unpdate_non_blank_token_index:
            A boolean flag to update non-blank token indices.
                When the blank token index is added, we may need to update the
                indices of non-blank tokens to make a room for the blank token
                index to avoid having conflicting indices. If this issue has
                been already taken care of, then set this flag to False. In
                fine tuning the Wav2Vec2.0 huggingface model, this flag needs
                to be set False.
    Returns:
        A dictionary containing a blank augmented sequence.
    """
    assert isinstance(inputs, dict)
    assert {"SEQ_DATA", "SEQ_LEN"} <= inputs.keys()

    # If some values are larger than blank_index, then those values are added
    # by one to make a room for the blank index.
    ids = torch.where(inputs["SEQ_DATA"] >= blank_index)
    updated_data = inputs["SEQ_DATA"].clone().detach()
    if update_non_blank_token_index:
        updated_data[ids] = inputs["SEQ_DATA"][ids] + 1

    output = {}
    # Creates a tensor filled with blank values.
    blank_tensor = torch.full(inputs["SEQ_DATA"].shape, fill_value=blank_index)

    # updated_data is interleaved with the blank tensor using "stacking" and
    # "reshaping".
    if boundary_blanks:
        data = torch.stack((blank_tensor, updated_data), axis=2)
        data = torch.reshape(data, (updated_data.shape[0], -1))

        # Concatenates a zero at the end of the sequence.
        padded = torch.full((updated_data.shape[0], 1), fill_value=blank_index)
        data = torch.concat((data, padded), axis=1)

        # If boundary_blanks are not used, then the length is 2 * L + 1.
        output["SEQ_LEN"] = 2 * inputs["SEQ_LEN"] + 1
    else:
        data = torch.stack((updated_data, blank_tensor), axis=2)
        data = torch.reshape(data, (updated_data.shape[0], -1))
        data = data[:, :-1]

        # If boundary_blanks are not used, then the length is 2 * L - 1.
        output["SEQ_LEN"] = 2 * inputs["SEQ_LEN"] - 1

    mask = sequence_mask(output["SEQ_LEN"],
                         maxlen=data.shape[1])
    output["SEQ_DATA"] = data * mask

    return output


def to_onset_augmented_labels(inputs: dict, num_classes: int) -> dict:
    assert isinstance(inputs, dict)
    assert {"SEQ_DATA", "SEQ_LEN"} <= inputs.keys()

    output = {}
    output["SEQ_LEN"] = 2 * inputs["SEQ_LEN"]

    in_data = 2 * inputs["SEQ_DATA"].clone().detach()

    data = torch.stack((in_data, in_data + 1), axis=2)
    data = torch.reshape(data, (inputs["SEQ_DATA"].shape[0], -1))
    mask = sequence_mask(output["SEQ_LEN"],
                         maxlen=data.shape[1])
    output["SEQ_DATA"] = data * mask

    return output


# Third-party imports
# * https://github.com/amaas/stanford-ctc/blob/master/ctc/ctc.py
# * https://github.com/HawkAaron/warp-transducer/blob/master/tensorflow_binding/src/warprnnt_op.cc
def calculate_log_label_prob(labels, softmax_output):
    """Calculates the log probability $\log(\hat{y}_{t, c_l})$.

    Computes the log probability of each label in the sequence $c_l$
    ($0 \le l < L$) predicted by the model at each time step $t$.
    The result is a 3D tensor where the value at $(b, t, l)$ corresponds
    to the batch index $b$, time index $t$, and label sequence index $l$.

    Args:
        labels: A tensor of shape (batch_size, max_labels_len) containing
            ground-truth label sequences. These sequences should already
            include blank labels.
        softmax_output: The model output tensor of shape
            (batch_size, max_seq_len, num_classes).

    Returns:
        A tensor of shape (batch_size, max_seq_len, max_labels_len)
        containing the calculated log probabilities.
    """
    max_logit_len = softmax_output.shape[1]
    labels = torch.tile(torch.unsqueeze(labels, dim=1), (1, max_logit_len, 1))
    return torch.log(torch.gather(input=softmax_output, dim=2, index=labels))


def _calculate_unnormalized_log_seq_prob(log_alpha, accum_log_seq_prob_sum,
                                         logit_len, label_len):
    # In alpha calculation, the log probabilty is normalized to prevent
    # over-flowing and under-flowing. This effect is compensated here.
    # log_p_ctc = log
    batch_size = log_alpha.shape[0]
    batch_index = torch.arange(batch_size, dtype=torch.int32)

    final_log_alpha0 = log_alpha[batch_index, logit_len - 1, label_len - 1]
    final_log_alpha1 = log_alpha[batch_index, logit_len - 1, label_len - 2]

    # max(alpha_{T-1,L-1}, alpha_{T-1,L})
    #
    # TODO(chanwcom)
    # There is an issue with the following statement.
    # It should be  addition rather than max.
    # alpha_{T-1,L-2}, alpha_{T-1,L-1}
    # log(Exp(log_alpha_{T-1, L-2}) + Exp(log_alpha_{T-1, L-1}))
    final_log_alpha = torch.max(final_log_alpha0, final_log_alpha1)

    # Finds the accumulated log seq probability at the last time index.
    final_accum = accum_log_seq_prob_sum[batch_index, logit_len - 1]

    return final_log_alpha + final_accum


class LabelType(enum.Enum):
    CTC = 0
    SHC = 1
    SHC_TYPE_0 = 2
    SHC_TYPE_1 = 3


class ThresholdType(enum.Enum):
    NO_THRESHOLD = 0
    ENTROPY = 1
    MAX_PROB = 2
    ELS = 3


class ProcessingType(enum.Enum):
    # Processing type is meaningful only when ENTROPY or MAX_PROB is selected
    # as the threshold type.
    UNCHANGED = 0
    UNIFORM = 1
    ZERO = 2


def label_trans_allowance_table_ctc(labels, labels_len):
    """Constructs a table containing the label transition allowance flags.

    We assume that label_seq should contain "blank labels" described in the
    original CTC paper.
    The shape of the returned tensor is (batch_size, max_seq_len, max_seq_len).
    The transition rule is as follows:

    Depending on whether the transition from the i-th label to the j-th label
    in the label sequence is allowed,
      a(b, i, j) = 0,         if this transition is allowed.
      a[b, i, j] = LOG_0:     if this transition is not allowed.

    Args:
        label_seq: A dictionary containing a batch of label sequences.
            * "DATA": A tensor containing label sequences.
                The shape is (batch_size, max_seq_length). Note that the data
                should follow the blank label rule, which states that "blank"
                labels should be interleaved with real labels. In addition to
                this, blank symbols are prepended and appended to the sequence.
            * "SEQ_LEN": A tensor containing the length of each label sequence.
                The shape is (batch_size).
    Returns:
        A tensor containing flags whether transitions are allowed.
            The shape is (batch_size, max_label_seq_len, max_seq_len)
    """

    max_seq_len = torch.max(labels_len)
    l = torch.arange(max_seq_len, dtype=torch.int32)

    # Indices corresponding to i -> i.
    indices0 = torch.stack([l, l], axis=1)

    # Indices corresponding to i -> i + 1.
    indices1 = torch.stack([l[:-1], l[:-1] + 1], axis=1)

    # Indices corresponding to i -> i + 2.
    indices2 = torch.stack([l[:-2], l[:-2] + 2], axis=1)

    # Constructs the transition table.
    indices = torch.concat([indices0, indices1, indices2], axis=0)
    values = torch.zeros([indices.shape[0]])

    trans_table = torch.full(size=(max_seq_len, max_seq_len), fill_value=LOG_0)
    trans_table[torch.unbind(indices, axis=1)] = 0

    batch_size = labels.shape[0]
    trans_table = torch.tile(torch.unsqueeze(trans_table, axis=0),
                             [batch_size, 1, 1])

    # Detects repeats and blank to blank transitions.
    #
    # These cases can be detected by checking whether y[l] == y[l + 2].
    indices = torch.where(labels[:, :-2] == labels[:, 2:])
    indices = (indices[0], indices[1], indices[1] + 2)
    trans_table[indices] = LOG_0

    return trans_table


def label_trans_table_shc_type0(labels, labels_len):
    """Constructs a table containing the label transition allowance flags.

    We assume that label_seq should contain "blank labels" described in the
    original CTC paper.
    The shape of the returned tensor is (batch_size, max_seq_len, max_seq_len).
    The transition rule is as follows:

    Depending on whether the transition from the i-th label to the j-th label
    in the label sequence is allowed,
      a[b, i, j] = 0,         if this transition is allowed.
      a[b, i, j] = LOG_0:     if this transition is not allowed.

    Args:
        label_seq: A dictionary containing a batch of label sequences.
            * "DATA": A tensor containing label sequences.
                The shape is (batch_size, max_seq_length). Note that the data
                should follow the blank label rule, which states that "blank"
                labels should be interleaved with real labels. In addition to
                this, blank symbols are prepended and appended to the sequence.
            * "SEQ_LEN": A tensor containing the length of each label sequence.
                The shape is (batch_size).
    Returns:
        A tensor containing flags whether transitions are allowed.
            The shape is (batch_size, max_label_seq_len, max_seq_len)
    """

    max_seq_len = torch.max(labels_len)
    l0 = torch.arange(1, max_seq_len, 2, dtype=torch.int32)
    l1 = torch.arange(max_seq_len, dtype=torch.int32)

    # Indices corresponding to i -> i.
    indices0 = torch.stack([l0, l0], axis=1)

    # Indices corresponding to i -> i + 1.
    indices1 = torch.stack([l1[:-1], l1[:-1] + 1], axis=1)

    # Constructs the transition table.
    indices = torch.concat([indices0, indices1], axis=0)
    values = torch.zeros([indices.shape[0]])

    trans_table = torch.full(size=(max_seq_len, max_seq_len), fill_value=LOG_0)
    trans_table[torch.unbind(indices, axis=1)] = 0

    batch_size = labels.shape[0]
    trans_table = torch.tile(torch.unsqueeze(trans_table, axis=0),
                             [batch_size, 1, 1])

    return trans_table

def label_trans_table_shc(labels, labels_len):
    """Constructs a table containing the label transition allowance flags.

    We assume that label_seq should contain "blank labels" described in the
    original CTC paper.
    The shape of the returned tensor is (batch_size, max_seq_len, max_seq_len).
    The transition rule is as follows:

    Depending on whether the transition from the i-th label to the j-th label
    in the label sequence is allowed,
      a[b, i, j] = 0,         if this transition is allowed.
      a[b, i, j] = LOG_0:     if this transition is not allowed.

    Args:
        label_seq: A dictionary containing a batch of label sequences.
            * "DATA": A tensor containing label sequences.
                The shape is (batch_size, max_seq_length). Note that the data
                should follow the blank label rule, which states that "blank"
                labels should be interleaved with real labels. In addition to
                this, blank symbols are prepended and appended to the sequence.
            * "SEQ_LEN": A tensor containing the length of each label sequence.
                The shape is (batch_size).
    Returns:
        A tensor containing flags whether transitions are allowed.
            The shape is (batch_size, max_label_seq_len, max_seq_len)
    """

    max_seq_len = torch.max(labels_len)
    l = torch.arange(max_seq_len, dtype=torch.int32)

    # Indices corresponding to i -> i.
    indices0 = torch.stack([l, l], axis=1)

    # Indices corresponding to i -> i + 1.
    indices1 = torch.stack([l[:-1], l[:-1] + 1], axis=1)

    # Constructs the transition table.
    indices = torch.concat([indices0, indices1], axis=0)
    values = torch.zeros([indices.shape[0]])

    trans_table = torch.full(size=(max_seq_len, max_seq_len), fill_value=LOG_0)
    trans_table[torch.unbind(indices, axis=1)] = 0

    batch_size = labels.shape[0]
    trans_table = torch.tile(torch.unsqueeze(trans_table, axis=0),
                             [batch_size, 1, 1])

    return trans_table

def label_trans_table_shc_type1(labels, labels_len):
    """Constructs a table containing the label transition allowance flags.

    We assume that label_seq should contain "blank labels" described in the
    original CTC paper.
    The shape of the returned tensor is (batch_size, max_seq_len, max_seq_len).
    The transition rule is as follows:

    Depending on whether the transition from the i-th label to the j-th label
    in the label sequence is allowed,
      a[b, i, j] = 0,         if this transition is allowed.
      a[b, i, j] = LOG_0:     if this transition is not allowed.

    Args:
        label_seq: A dictionary containing a batch of label sequences.
            * "DATA": A tensor containing label sequences.
                The shape is (batch_size, max_seq_length). Note that the data
                should follow the blank label rule, which states that "blank"
                labels should be interleaved with real labels. In addition to
                this, blank symbols are prepended and appended to the sequence.
            * "SEQ_LEN": A tensor containing the length of each label sequence.
                The shape is (batch_size).
    Returns:
        A tensor containing flags whether transitions are allowed.
            The shape is (batch_size, max_label_seq_len, max_seq_len)
    """

    max_seq_len = torch.max(labels_len)
    l = torch.arange(max_seq_len, dtype=torch.int32)

    # Indices corresponding to i -> i.
    indices0 = torch.stack([l, l], axis=1)

    # Indices corresponding to i -> i + 1.
    indices1 = torch.stack([l[:-1], l[:-1] + 1], axis=1)

    # Constructs the transition table.
    indices = torch.concat([indices0, indices1], axis=0)
    values = torch.zeros([indices.shape[0]])

    trans_table = torch.full(size=(max_seq_len, max_seq_len), fill_value=LOG_0)
    trans_table[torch.unbind(indices, axis=1)] = 0

    batch_size = labels.shape[0]
    trans_table = torch.tile(torch.unsqueeze(trans_table, axis=0),
                             [batch_size, 1, 1])

    return trans_table


# TODO TODO(chanwcom)
# Refactor as a class
def label_trans_allowance_table(labels, label_len, label_type: LabelType):
    if label_type == LabelType.CTC:
        table = label_trans_allowance_table_ctc(labels, label_len)
    elif label_type == LabelType.SHC:
        table = label_trans_table_shc(labels, label_len)
    elif label_type == LabelType.SHC_TYPE_0:
        table = label_trans_table_shc_type0(labels, label_len)
    elif label_type == LabelType.SHC_TYPE_1:
        table = label_trans_table_shc_type1(labels, label_len)
    else:
        raise ValueError("Unsupported type.")

    return table


class CtcLoss(torch.autograd.Function):
    """A class for calculating the CTC loss."""

    @staticmethod
    def forward(ctx,
                labels,
                labels_len,
                logits,
                logits_len,
                label_type: LabelType = LabelType.CTC,
                update_non_blank_token_index: bool = True,
                threshold_type: ThresholdType = ThresholdType.NO_THRESHOLD,
                threshold: float = 0.1,
                processing_type: ProcessingType = ProcessingType.UNCHANGED):
        """Calculates the Connectionist Temporal Classification (CTC) loss.

        Args:
            ctx: Contexts for this CtcLoss operation.
            labels: A tensor containing batch of ground-truth label sequences.
                Note that this label sequence should already include blank labels.
                The shape is given by (batch_size, max_labels_len).
            labels_len: The lengths of labels that has the shape of
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

        # Converting the sequences.
        # Note that the following is only for HuggingFace case.
        # In case of HuggingFace, the boundary blanks should be added and non
        # -blank token indices should NOT be updated.

        inputs = {}
        inputs["SEQ_DATA"] = labels
        inputs["SEQ_LEN"] = labels_len
        if label_type == LabelType.CTC:
            inputs = to_blank_augmented_labels(inputs, 0, True,
                                               update_non_blank_token_index)
            # TODO  TODO(chanwcom )The following is the correc one.
            #inputs = to_blank_augmented_labels(inputs, 0, True, False)
        elif (label_type == LabelType.SHC_TYPE_0
              or label_type == LabelType.SHC_TYPE_1):
            raise NotImplementedError
            # How to find num_classes?
            # It is not easy for Hugging face fine tuning.
            #inputs =  to_onset_augmented_labels(inputs, num_classes)
        else:
            raise ValueEror("Unsupported label sequence format type.")
        labels = inputs["SEQ_DATA"]
        labels_len = inputs["SEQ_LEN"]

        log_label_prob = calculate_log_label_prob(
            labels, torch.softmax(logits, dim=-1))

        trans_table = label_trans_allowance_table(labels, labels_len,
                                                  label_type)

        # Alpha and beta should be calculated.
        log_alpha, log_beta, log_seq_prob = calculate_alpha_beta(
            trans_table, log_label_prob, labels_len, logits_len)

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
        # The shape of log_gamma is (batch_size, max_logits_len, max_label_len).
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - torch.logsumexp(
            log_gamma, axis=2, keepdim=True)

        # To ignore an invalid loss case.
        #
        # If labels_len < logits_len, then the loss is not valid.
        invalid_length_mask = (torch.greater_equal(
            logits_len, labels_len)).type(torch.float32)

        loss = -torch.multiply(log_seq_prob, invalid_length_mask)

        max_label_len = torch.max(labels_len)
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
        for l in range(max_label_len):
            onehot = (1.0 - (torch.nn.functional.one_hot(
                labels[:, l], num_classes))) * LOG_0

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

        if threshold_type != ThresholdType.NO_THRESHOLD:
            if processing_type == ProcessingType.UNIFORM:
                uniform_flag = True
            else:
                uniform_flag = False

            ground_truth_prob, flag = apply_postprocessing(
                ground_truth_prob, logits_len, threshold_type, threshold,
                uniform_flag)

        gradient = -(ground_truth_prob - torch.softmax(logits, dim=2))

        if (threshold_type != ThresholdType.NO_THRESHOLD
                and processing_type == ProcessingType.ZERO):

            gradient = torch.multiply(gradient, flag)

        # To ignore an invalid loss case.
        #
        # If labels_len < logits_len, then the loss is not valid.
        invalid_length_mask = (torch.greater_equal(
            logits_len, labels_len)).type(torch.float32)
        gradient = torch.multiply(
            gradient, torch.reshape(invalid_length_mask, (-1, 1, 1)))

        # Seqeunce mask
        seq_mask = sequence_mask(logits_len,
                                 maxlen=torch.max(logits_len))

        # The dimension of "gradient" is (batch_size, logit_len, num_classes)
        gradient = torch.multiply(gradient, torch.unsqueeze(seq_mask, axis=2))

        ctx.save_for_backward(gradient)

        return loss

    @staticmethod
    def backward(ctx, grad):
        gradient, = ctx.saved_tensors
        gradient = torch.multiply(gradient, torch.reshape(grad, (-1, 1, 1)))

        return None, None, gradient, None, None, None, None, None, None

@torch.jit.script
def calculate_alpha_beta(label_trans_table: torch.Tensor, 
                         log_label_prob: torch.Tensor, 
                         label_len: torch.Tensor,
                         logit_len: torch.Tensor):
    """Calculates alpha and beta variables for CTC computation.

    This function implements the forward-backward algorithm to compute 
    log_alpha and log_beta. It uses normalization at each step to ensure 
    numerical stability.

    Args:
        label_trans_table: Transition probabilities of shape (B, L, L).
        log_label_prob: Log probabilities of labels of shape (B, T, L).
        label_len: Actual lengths of label sequences of shape (B).
        logit_len: Actual lengths of logit sequences of shape (B).
        LOG_0: A constant representing log(0), typically a large negative number.

    Returns:
        log_alpha: Computed forward variables.
        log_beta: Computed backward variables.
        log_seq_prob_final: The final unnormalized sequence log probabilities.
    """
    LOG_0 = -706.893623  # float(np.log(1e-307))

    batch_size = log_label_prob.shape[0]
    max_label_len = int(torch.max(label_len))
    max_logit_len = int(torch.max(logit_len))
    device = log_label_prob.device

    # Initialize log_alpha and log_beta.
    log_alpha = torch.full((batch_size, max_logit_len, max_label_len),
                           fill_value=LOG_0, device=device)
    log_beta = torch.full((batch_size, max_logit_len, max_label_len),
                          fill_value=LOG_0, device=device)

    # 1. Forward Pass (log_alpha)
    prev_log_alpha = torch.full((batch_size, max_label_len), 
                                fill_value=LOG_0, device=device)
    prev_log_alpha[:, 0] = 0.0
    alpha_max_list = []

    for t in range(max_logit_len):
        transitions = prev_log_alpha.unsqueeze(2) + label_trans_table
        log_alpha_t = torch.logsumexp(transitions, dim=1) + log_label_prob[:, t, :]

        log_alpha_max = torch.max(log_alpha_t, dim=1, keepdim=True).values
        log_alpha_t -= log_alpha_max

        log_alpha[:, t, :] = log_alpha_t
        prev_log_alpha = log_alpha_t
        alpha_max_list.append(log_alpha_max.squeeze(-1))

    accum_log_alpha_max = torch.cumsum(torch.stack(alpha_max_list, dim=1), dim=1)

    # 2. Backward Pass (log_beta)
    # Optimized initial_log_beta creation.
    initial_log_beta = torch.full((batch_size, max_label_len), 
                                  fill_value=LOG_0, device=device)
    for i in range(batch_size):
        initial_log_beta[i, label_len[i] - 1] = 0.0

    prev_log_beta = initial_log_beta

    time_mask = sequence_mask(logit_len, max_logit_len).unsqueeze(2).to(torch.bool)

    next_log_label_prob = torch.zeros((batch_size, max_label_len), device=device)

    for t in range(max_logit_len - 1, -1, -1):
        messages = (prev_log_beta + next_log_label_prob).unsqueeze(1)
        log_beta_t = torch.logsumexp(messages + label_trans_table, dim=2)
        
        # Current label probs will be 'next' for the step t-1.
        next_log_label_prob = log_label_prob[:, t, :]

        # Masking and normalization.
        log_beta_t = torch.where(time_mask[:, t, :], log_beta_t, initial_log_beta)
        log_beta_max = torch.max(log_beta_t, dim=1, keepdim=True).values
        log_beta_t -= log_beta_max

        log_beta[:, t, :] = log_beta_t
        prev_log_beta = log_beta_t

    # 3. Post-processing and Final Sequence Probability
    # Apply time mask and label mask for clean output.
    log_alpha.masked_fill_(~time_mask, LOG_0)
    log_beta.masked_fill_(~time_mask, LOG_0)

    label_mask = sequence_mask(label_len, max_label_len).unsqueeze(1).to(torch.bool)
    log_alpha.masked_fill_(~label_mask, LOG_0)
    log_beta.masked_fill_(~label_mask, LOG_0)

    log_seq_prob_final = _calculate_unnormalized_log_seq_prob(
        log_alpha, accum_log_alpha_max, logit_len, label_len)

    return log_alpha, log_beta, log_seq_prob_final



def apply_postprocessing(ground_truth_prob: torch.Tensor,
                         logits_len: torch.Tensor,
                         threshold_type: ThresholdType,
                         threshold: float,
                         uniform: bool = True):

    if threshold_type == ThresholdType.NO_THRESHOLD:
        return ground_truth_prob
    elif threshold_type == ThresholdType.ELS:
        return _apply_smoothing(ground_truth_prob, threshold)
    elif threshold_type == ThresholdType.ENTROPY:
        flag = (torch.sum(torch.special.entr(ground_truth_prob), axis=2)
                <= threshold)
    elif threshold_type == ThresholdType.MAX_PROB:
        flag = torch.max(ground_truth_prob, axis=2).values >= threshold
    else:
        raise ValueError("Unsupported threshold type.")
    flag = torch.unsqueeze(flag, 2).type(torch.float32)

    # Makes the one hot representation based on argmax.
    one_hot = torch.nn.functional.one_hot(
        torch.argmax(ground_truth_prob, dim=2),
        ground_truth_prob.shape[2]).type(torch.float32)

    if uniform:
        others = torch.ones_like(
            ground_truth_prob) / ground_truth_prob.shape[2]
    else:
        others = ground_truth_prob

    mask = torch.unsqueeze(
        sequence_mask(logits_len,
                      ground_truth_prob.shape[1]), axis=2) # yapf: disable

    return ((one_hot * flag + (1 - flag) * others) * mask, flag)


def _apply_smoothing(ground_truth_prob: torch.Tensor, smoothing_coeff: float):
    """Applies smoothing using the ELS algorithm.

    Args:
        ground_truth_prob: A tensor containing exp(alaph+beta)
            The shape is (batch_size, logit_length, num_classes)
            Note that "ground_truth_prob" does not contain log probabilities
            but original probabilities that take values between 0 and 1.
        smoothing_coeff:

    Returns:
    """
    if smoothing_coeff == 0.0:
        return ground_truth_prob

    # Finds cases of an one-hot vector.
    #
    # In this case, division by zero error will occur if we do not pre-process
    # it. So temporarily add these vectors  with 1e-10.
    ids = torch.where(ground_truth_prob == 1)

    outputs = ground_truth_prob.clone().detach()

    # It has the effect of adding a small value to the entire [i, t, :] so
    # that division by zero will not happen.
    outputs[ids[0], ids[1], :] = 1e-10

    # Values higher than 1 - (smoothing_coeff) are replaced with zeros.
    ids_too_large = torch.where(ground_truth_prob > 1 - smoothing_coeff)
    wo_largest = outputs.clone().detach()
    wo_largest[ids_too_large] = 0.0

    # Obtains the sum without the largest values.
    sum_wo_largest = torch.sum(wo_largest, axis=2)

    # Finds the maximum value along the label axis. Subtracts this value by
    # (1 - smoothing coeff) and floors by zero.
    #
    # This will be the amount that will be subtracted from the maximum value
    # and subsequently added to other values belonging to the same time step.

    smoothing_values = (torch.maximum(
        torch.max(ground_truth_prob, axis=2).values - (1 - smoothing_coeff),
        torch.Tensor([0.0]).to(ground_truth_prob.device)))

    scaling_coeff = torch.div(
        smoothing_values + sum_wo_largest,
        torch.maximum(sum_wo_largest, torch.Tensor([1e-10]).to(ground_truth_prob.device)))

    outputs = torch.unsqueeze(scaling_coeff, dim=-1) * outputs

    # Replaces too large values with 1.0-smoothing_coeff.
    outputs[ids_too_large] = 1.0 - smoothing_coeff

    return outputs
