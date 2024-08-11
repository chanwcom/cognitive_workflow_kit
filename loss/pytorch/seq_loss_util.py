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

LOG_0 = torch.tensor(np.log(1e-307)).type(torch.float32)


def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    """Applies sequence masking.

    This implementation is based on the following website.
    https://discuss.pytorch.org/t/pytorch-equivalent-for-tf-sequence-mask/39036/3

    """
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    return mask.type(dtype)

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
                         maxlen=data.shape[1],
                         dtype=data.dtype)
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
                         maxlen=data.shape[1],
                         dtype=data.dtype)
    output["SEQ_DATA"] = data * mask

    return output


# Third-party imports
# * https://github.com/amaas/stanford-ctc/blob/master/ctc/ctc.py
# * https://github.com/HawkAaron/warp-transducer/blob/master/tensorflow_binding/src/warprnnt_op.cc
def calculate_log_label_prob(labels, softmax_output):
    """Calculates log({\hat{y}_t}_{c_l}).

    This calculates the log probability of each label in the label sequence
    c_l 0 <= l <= L-1 predicted by the model at time t. The returned value is
    a three-dimensional tensor, where value is stored in (b, t, l) where b is
    the batch index, t is the time index, and l ls the label sequence index.

    Args:
        labels: A tensor containing a batch of ground-truth label sequences.
            Note that this label sequence should already include blank labels.
            The shape is given by (batch_size, max_labels_len).
        softmax_output: The output of the model.
            The shape is given by:
            (batch_size, max_seq_len, num_classes).

    Returns:
        The shape is (batch, max_logit_len, max_labels_len).
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
    SHC_TYPE_0 = 1
    SHC_TYPE_1 = 2


class ThresholdType(enum.Enum):
    NO_THRESHOLD = 0
    ENTROPY = 1
    MAX_PROB = 2


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
    indices = [indices[0], indices[1], indices[1] + 2]
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
def label_trans_allowance_table(labels, label_len, table_type: LabelType):
    if table_type == LabelType.CTC:
        table = label_trans_allowance_table_ctc(labels, label_len)
    elif table_type == LabelType.SHC_TYPE_0:
        table = label_trans_table_shc_type0(labels, label_len)
    elif table_type == LabelType.SHC_TYPE_1:
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
                table_type: LabelType = LabelType.CTC,
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
        if table_type == LabelType.CTC:
            inputs = to_blank_augmented_labels(inputs, 0, True,
                                               update_non_blank_token_index)
            # TODO  TODO(chanwcom )The following is the correc one.
            #inputs = to_blank_augmented_labels(inputs, 0, True, False)
        elif (table_type == LabelType.SHC_TYPE_0
              or table_type == LabelType.SHC_TYPE_1):
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
                                                  table_type)

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

        # TODO TODO(chanwcom)
        # Post processing needs to be done here.
        #processing_type: ProcessingType = ProcessingType.UNCHANGED,
        #uniform_flag: bool = True):

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
                                 maxlen=torch.max(logits_len),
                                 dtype=log_gamma.dtype)

        # The dimension of "gradient" is (batch_size, logit_len, num_classes)
        gradient = torch.multiply(gradient, torch.unsqueeze(seq_mask, axis=2))

        ctx.save_for_backward(gradient)

        return loss

    @staticmethod
    def backward(ctx, grad):
        gradient, = ctx.saved_tensors
        gradient = torch.multiply(gradient, torch.reshape(grad, (-1, 1, 1)))

        return None, None, gradient, None, None, None, None, None, None


def calculate_alpha_beta(label_trans_table, log_label_prob, label_len,
                         logit_len):
    """Calculates the alpha best and beta best variables.

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
    mask = sequence_mask(logit_len, maxlen=max_logit_len)

    prev_log_alpha = ((1.0 - (torch.nn.functional.one_hot(
        torch.zeros(size=(batch_size, ), dtype=torch.int64), max_label_len))) *
                      LOG_0)
    accum_log_alpha_max = torch.zeros((batch_size, max_logit_len),
                                      dtype=torch.float32)
    prev_log_alpha_max = torch.zeros((batch_size), dtype=torch.float32)

    for t in range(max_logit_len):
        # Calculates log_alpha recursively from the previous time step.
        log_alpha[:, t, :] = (
            torch.logsumexp(
                torch.add(torch.unsqueeze(prev_log_alpha, axis=2),
                          label_trans_table),
                dim=1) + log_label_prob[:, t, :]) # yapf: disable

        # Normalizes the log sequence prob.
        log_alpha_max = torch.max(log_alpha[:, t, :], axis=1,
                                  keepdims=True).values
        log_alpha[:, t, :] -= log_alpha_max

        # Accumulates the maximum.
        accum_log_alpha_max[:, t] = (prev_log_alpha_max +
                                     torch.squeeze(log_alpha_max, axis=-1))
        prev_log_alpha_max = accum_log_alpha_max[:, t]
        prev_log_alpha = log_alpha[:, t, :]

    initial_log_beta = (
        (1.0 - torch.nn.functional.one_hot(label_len - 1, max_label_len)) *
        LOG_0)
    prev_log_beta = initial_log_beta

    time_mask = torch.unsqueeze(
        sequence_mask(logit_len, maxlen=max_logit_len, dtype=torch.float32),
        axis=2) # yapf: disable

    next_log_label_prob = torch.zeros(size=(batch_size, max_label_len))
    for t in range(max_logit_len - 1, -1, -1):
        # Calculates log_beta recursively from the next time step.
        log_beta[:, t, :] = (
            torch.logsumexp(
                torch.add(torch.unsqueeze(
                    prev_log_beta + next_log_label_prob, 1),
                    label_trans_table),
                dim=2)) # yapf: disable

        next_log_label_prob = log_label_prob[:, t, :]

        # Normalizes the log beta prob. using the maximum value at time t.
        log_beta_max = torch.max(log_beta[:, t, :], axis=1,
                                 keepdims=True).values
        log_beta[:, t, :] -= log_beta_max

        # Correctly initializes log_beta from the length info.
        #
        # If mask is zero, then makes the current log_beta zero
        # first multiplying with the mask. After that, re-initializes the
        # log_beta to be "initial_log_beta".

        log_beta[:, t, :] = torch.multiply(log_beta[:, t, :],
                                           time_mask[:, t, :]) # yapf: disable
        log_beta[:, t, :] += torch.multiply(initial_log_beta,
                                            (1.0 - time_mask[:, t, :]))

        prev_log_beta = log_beta[:, t, :]

    log_alpha += torch.multiply(LOG_0, (1.0 - time_mask))
    log_beta += torch.multiply(LOG_0, (1.0 - time_mask))

    label_mask = torch.unsqueeze(sequence_mask(label_len,
                                               maxlen=max_label_len,
                                               dtype=torch.float32),
                                 axis=1)
    log_alpha += torch.multiply(LOG_0, (1.0 - label_mask))
    log_beta += torch.multiply(LOG_0, (1.0 - label_mask))

    # We utilize the "tf.stop_gradient" API with the "tf.nest.map_structure"
    # API based on the recommendation in the following page:
    # https://www.tensorflow.org/api_docs/python/tf/scan

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
    elif threshold_type == ThresholdType.ENTROPY:
        flag = ((torch.sum(torch.special.entr(ground_truth_prob), axis=2) /
                 torch.log(torch.tensor(ground_truth_prob.shape[2])))
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
                      ground_truth_prob.shape[1],
                      dtype=ground_truth_prob.dtype), axis=2) # yapf: disable

    return ((one_hot * flag + (1 - flag) * others) * mask, flag)
