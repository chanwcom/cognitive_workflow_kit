"""A module implementing utilities for sequence losses."""

# pylint: disable=no-member, invalid-name, import-error

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Third-party imports
import tensorflow as tf
import tensorflow_probability as tfp

log_0 = tf.cast(tf.math.log(tf.cast(0, tf.dtypes.float64) + 1e-307),
                tf.dtypes.float32)
#LOG_0 = tf.constant(np.log(np.finfo(np.float64).tiny).astype(np.float32))

def to_blank_augmented_labels(
        inputs: dict, blank_index: int=0,
        boundary_blanks: bool=True) -> dict:  # yapf: disable
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
    Returns:
        A dictionary containing a blank augmented sequence.
    """
    assert isinstance(inputs, dict)
    assert {"SEQ_DATA", "SEQ_LEN"} <= inputs.keys()

    # If some values are larger than blank_index, then those values are added
    # by one to make a room for the blank index.
    ids = tf.where(inputs["SEQ_DATA"] >= blank_index)
    values = tf.gather_nd(inputs["SEQ_DATA"], ids) + 1
    updated_data = tf.tensor_scatter_nd_update(inputs["SEQ_DATA"], ids, values)

    output = {}

    # Creates a tensor filled with blank values.
    blank_tensor = tf.fill(tf.shape(inputs["SEQ_DATA"]), blank_index)

    # updated_data is interleaved with the blank tensor using "stacking" and
    # "reshaping".
    if boundary_blanks:
        data = tf.stack((blank_tensor, updated_data), axis=2)
        data = tf.reshape(data, (tf.shape(updated_data)[0], -1))

        # Concatenates a zero at the end of the sequence.
        padded = tf.fill((tf.shape(updated_data)[0], 1), blank_index)
        data = tf.concat((data, padded), axis=1)

        # If boundary_blanks are not used, then the length is 2 * L + 1.
        output["SEQ_LEN"] = 2 * inputs["SEQ_LEN"] + 1
    else:
        data = tf.stack((updated_data, blank_tensor), axis=2)
        data = tf.reshape(data, (tf.shape(updated_data)[0], -1))
        data = data[:, :-1]

        # If boundary_blanks are not used, then the length is 2 * L - 1.
        output["SEQ_LEN"] = 2 * inputs["SEQ_LEN"] - 1

    mask = tf.cast(tf.sequence_mask(output["SEQ_LEN"],
                                    maxlen=tf.shape(data)[1]),
                   dtype=data.dtype)
    output["SEQ_DATA"] = data * mask

    return output


def calculate_initial_log_seq_prob(label_len):
    """Calculates CTC alignment initial and final state log probabilities.

    Create the initial/final state values directly as log values to avoid
    having to take a float64 log on tpu (which does not exist).

    Args:
        label_len: int tensor of shape [batch_size], seq lengths in the batch.
        max_label_len: int, max sequence length possible.

    Returns:
        initial_state_log_probs, final_state_log_probs
    """

    batch_size = _get_dim(label_len, 0)
    max_label_len = tf.math.reduce_max(label_len)
    initial_forward_log_seq_prob = tf.one_hot(indices=tf.zeros(
        [batch_size], dtype=tf.dtypes.int32),
                                              depth=max_label_len,
                                              on_value=0.0,
                                              off_value=log_0,
                                              axis=1)

    initial_backward_log_seq_prob = tf.one_hot(indices=label_len - 1,
                                               depth=max_label_len,
                                               on_value=0.0,
                                               off_value=log_0,
                                               axis=1)

    return initial_forward_log_seq_prob, initial_backward_log_seq_prob


# Third-party imports
# * https://github.com/amaas/stanford-ctc/blob/master/ctc/ctc.py
# * https://github.com/HawkAaron/warp-transducer/blob/master/tensorflow_binding/src/warprnnt_op.cc
def calculate_log_label_prob(labels, softmax_output):
    """Calculates log((\hat{y}_t)_{c_l}).

    The returned value is a three-dimensional tensor, where value is stored
    in (b, t, l) where b is the batch index, t is the time index, and l ls the
    label sequence index.

    Args:
        labels: A tensor containing a batch of ground-truth label sequences.
            Note that this label sequence should already include blank labels.
            The shape is given by (batch_size, max_labels_len).
        softmax_output: The output of the model.
            The shape is given by:
            (batch_size, max_seq_len, num_classes).

    Returns:
        The shape is (batch, max_seq_len, max_label_len).
    """
    return tf.math.log(tf.gather(softmax_output, labels, batch_dims=1, axis=2))


def _calculate_unnormalized_log_seq_prob(gamma, accum_log_seq_prob_sum,
                                         logit_len, label_len):
    batch_size = _get_dim(gamma, 0)
    batch_range = tf.range(batch_size)

    # Obtains the list of indices for the sequence ends.
    #
    # Note that the sequence may end at L -1 (blank label) or L - 2 (the last
    # non-blank label).
    indices0 = tf.stack([batch_range, logit_len - 1, label_len - 1], axis=1)
    indices1 = tf.stack([batch_range, logit_len - 1, label_len - 2], axis=1)
    indices = tf.concat([indices0, indices1], axis=0)

    seq_final_value = tf.transpose(
        tf.reshape(tf.gather_nd(gamma, indices), shape=[-1, batch_size]))
    seq_final_value = tf.reduce_max(seq_final_value, axis=1)

    # Finds the accumulated log seq probability at the last time index.
    indices = tf.stack([batch_range, logit_len - 1], axis=1)
    final_accum = tf.gather_nd(accum_log_seq_prob_sum, indices)

    return seq_final_value + final_accum


def label_trans_table(labels, labels_len):
    """Constructs a table containing the label transition flags.

    We assume that label_seq should contain "blank labels" described in the
    original CTC paper.
    The shape of the returned tensor is (batch_size, max_seq_len, max_seq_len).
    The transition rule is as follows:

    Depending on whether the transition from the i-th label to the j-th label
    in the label sequence is allowed,
      a(b, i, j) = 0,         if this transition is allowed.
      a[b, i, j] = log_0:     if this transition is not allowed.

    Args:
        label_seq: A dictionary containing a batch of label sequences.
            * "DATA": A tensor containing label sequences.
                The shape is (batch_size, max_seq_length). Note that the data
                should follow the blank label rule, which states that "blank"
                labels should be interleaved with real labels. In addition to
                this, blank symbols are prepended and appended to the sequence.
            " "SEQ_LEN": A tensor containing the length of each label sequence.
                The shape is (batch_size).
    Returns:
        A tensor containing flags whether transitions are allowed.
            The shape is (batch_size, max_label_seq_len, max_seq_len)
    """

    max_seq_len = tf.math.reduce_max(labels_len)
    l = tf.range(max_seq_len)

    # Indices corresponding to i -> i.
    indices0 = tf.stack([l, l], axis=1)

    # Indices corresponding to i -> i + 1.
    indices1 = tf.stack([l[:-1], l[:-1] + 1], axis=1)

    # Indices corresponding to i -> i + 2.
    indices2 = tf.stack([l[:-2], l[:-2] + 2], axis=1)

    # Constructs the transition table.
    indices = tf.concat([indices0, indices1, indices2], axis=0)
    values = tf.zeros([_get_dim(indices, 0)])

    trans_table = tf.fill(dims=(max_seq_len, max_seq_len), value=log_0)
    trans_table = tf.tensor_scatter_nd_update(trans_table, indices, values)

    data = labels
    batch_size = _get_dim(data, 0)

    trans_table = tf.tile(tf.expand_dims(trans_table, axis=0),
                          [batch_size, 1, 1])

    # Detects repeats and blank to blank transitions.
    #
    # These cases can be detected by checking whether y[l] == y[l + 2].
    indices = tf.where(tf.math.equal(data[:, :-2], data[:, 2:]))
    indices = tf.concat(
        [indices, tf.expand_dims(indices[:, 1] + 2, axis=1)], axis=1)
    values = tf.fill(dims=[_get_dim(indices, 0)], value=log_0)

    trans_table = tf.tensor_scatter_nd_update(trans_table, indices, values)

    return trans_table


@tf.custom_gradient
def ctc_loss(labels, labels_len, logits, logits_len):
    """Calculates the Connectionist Temporal Classification (CTC) loss.

    Args:
        labels: A tensor containing batch of ground-truth label sequences.
            Note that this label sequence should already include blank labels.
            The shape is given by (batch_size, max_labels_len).
        labels_len: The lengths of labels that has the shape of
            (batch_size).
        logits: The predicted "logit value". The shape is given by
            (batch_size, max_logit_seq_len, num_classes).
        logits_len: The len of logits that has the shape of (batch_size).
        smoothing_coeff:
        apply_smoothing_th:

    Note that zero values are assumed to be masked-values.

    Returns:
        A tuple containing (loss, grad)
    """
    # Note that the shape of labels is (B, L).
    tf.debugging.assert_equal(tf.rank(labels), 2)
    # Note that the shape of logits is (B, M, C).
    tf.debugging.assert_equal(tf.rank(logits), 3)
    # Checks the batch size.
    tf.debugging.assert_equal(tf.shape(labels)[0], tf.shape(logits)[0])

    log_label_prob = calculate_log_label_prob(labels,
                                              tf.nn.softmax(logits, axis=-1))

    trans_table = label_trans_table(labels, labels_len)

    alpha, beta, log_seq_prob = calculate_alpha_beta(trans_table,
                                                     log_label_prob,
                                                     labels_len, logits_len)

    # To ignore an invalid loss case.
    #
    # If labels_len < logits_len, then the loss is not valid.
    invalid_length_mask = tf.dtypes.cast(tf.math.greater_equal(
        tf.cast(logits_len, dtype=tf.dtypes.int64),
        tf.cast(labels_len, dtype=tf.dtypes.int64)),
                                         dtype=tf.float32)

    loss = -tf.math.multiply_no_nan(log_seq_prob, invalid_length_mask)

    # "gamma" is the posterior probability of label index.
    #
    # log_gamma is defined as follows:
    #   log p(q_{[m]} = l'| x, y) where m is the temporal index, and l' is the
    # blank-augmented label index. q_{[m]} is the "path variable" at time m.
    # The shape of log_gamma is (batch_size, max_logits__len, max_label_len).
    log_gamma = alpha + beta
    gamma_sum = tf.math.reduce_logsumexp(log_gamma, axis=2, keepdims=True)
    log_gamma -= gamma_sum

    # \tau is the average of the maximum posterior label index probability.
    #
    # This will be employed as a threshold deciding whether smoothing will be
    # applied or not.
    seq_mask = tf.cast(
        tf.sequence_mask(
            logits_len, maxlen=tf.math.reduce_max(logits_len)),
        dtype=log_gamma.dtype) # yapf: disable

    tau = tf.math.divide_no_nan(
        tf.math.reduce_sum(
            tf.math.reduce_max(tf.math.exp(log_gamma), axis=2) * seq_mask,
            axis=1),
        tf.cast(logits_len, dtype=tf.dtypes.float32)) # yapf: disable

    def grad(upstream):
        vocab_size = _get_dim(logits, 2)

        nominator = tf.fill(dims=tf.shape(logits), value=log_0)
        max_labels_len = tf.math.reduce_max(labels_len)

        # Update is done for each label to reduce memory requirement.
        def while_body(l, nominator):
            xi_l = log_gamma[:, :, l]
            onehot = tf.one_hot(labels[:, l],
                                depth=vocab_size,
                                on_value=0.0,
                                off_value=log_0)
            updates = tf.expand_dims(xi_l, axis=2) + tf.expand_dims(onehot,
                                                                    axis=1)

            nominator = tfp.math.log_add_exp(nominator, updates)

            return (l + 1, nominator)

        l = tf.constant(0)
        nominator = tf.while_loop(
            lambda l, _1: tf.less(l, max_labels_len),
            while_body, [l, nominator])[1] # yapf: disable

        exp_nominator = tf.math.exp(nominator)
        # Since log_gamma is already normalized with respect to all the labels, just
        # subtracting the nominator is fine.
        #
        #  if smoothing_flag is one, then smoothing result will be used.
        #  Otherwise,the original exp_nominator will be used in calculating the
        #  gradient.
        gradient = (tf.nn.softmax(logits, axis=2) - exp_nominator)

        mask = tf.reshape(invalid_length_mask, (-1, 1, 1))
        gradient = tf.math.multiply(gradient, mask)

        # The shape of gradient is (batch_size, max_logits_seq_len, num_classes).
        gradient = tf.math.multiply(gradient, tf.expand_dims(seq_mask, axis=2))
        gradient = tf.math.multiply(gradient, tf.reshape(upstream, (-1, 1, 1)))

        return [None, None, gradient, None]

    return loss, grad


def _get_dim(tensor, i):
    """Get value of tensor shape[i] preferring static value if available."""
    return tf.compat.dimension_value(tensor.shape[i]) or tf.shape(tensor)[i]


def calculate_alpha_beta(label_trans_table, log_label_prob, label_len,
                         logit_len):
    """Calculates the alpha best and beta best variables.

    This calculates the alpha and beta variables required for CTC computation.
    Note that the definition of beta variable is somewhat different from the
    original CTC paper. This equation will be explained in my future paper.
    TODO(chanwcom) Adds the paper link.

    Args:
        label_trans_table: A tensor containing the transition tables.
            The shape is (batch_size, max_label_len, max_label_len).
        log_label_prob: A tensor of posterior probabilities of each label.
            The shape is (batch_size, max_logit_len, max_label_len).
            Mathematically, it is given by the following equation:
                log (p_{[m]}(y_l | x)).
        label_len: A tensor containing the label lengths.
            The shape is (batch_size).
        logit_len: A tensor containing the logit lengths.
            The shape is (batch_size).

    Returns:
        A tuple of (alpha, beta, log_seq_prob_final)
        alpha: A tensor containing the "normalized" forward variable.
            The shape is (batch_size, max_logit_len, max_label_len).
        beta: A tensor containing the "normazlied" backward variable.
            The shape is (batch_size, max_logit_len, max_label_len).
        log_seq_prob_final:

    """
    batch_size = _get_dim(log_label_prob, 0)
    max_label_len = tf.math.reduce_max(label_len)
    max_logit_len = tf.math.reduce_max(logit_len)

    initial_log_seq_probs = calculate_initial_log_seq_prob(label_len)

    def forward(accum, elem):
        log_seq_prob, accum_log_seq_prob_sum = accum
        log_label_prob, mask = elem

        log_seq_prob = tf.expand_dims(log_seq_prob, axis=-1)

        # Finds the maximum log_seq_prob. from the previous time step.
        log_seq_prob = tf.math.reduce_logsumexp(
            tf.math.add(log_seq_prob, label_trans_table),
            axis=1) # yapf: disable

        # The log label probabilities are added.
        log_seq_prob += log_label_prob

        # Normalizes the log sequence prob.
        log_seq_prob_sum = tf.math.reduce_max(log_seq_prob,
                                              axis=1,
                                              keepdims=True)
        log_seq_prob -= log_seq_prob_sum

        # Accumulates the sum.
        accum_log_seq_prob_sum += tf.squeeze(log_seq_prob_sum, axis=-1)

        return (log_seq_prob, accum_log_seq_prob_sum)

    def backward(log_seq_prob, elem):
        log_label_prob, mask = elem

        log_seq_prob += log_label_prob

        log_seq_prob = tf.expand_dims(log_seq_prob, axis=1)
        log_seq_prob = tf.math.reduce_logsumexp(
            tf.math.add(log_seq_prob, label_trans_table), axis=2) # yapf: disable

        # Normalizes the log sequence prob.
        log_seq_prob_sum = tf.math.reduce_max(log_seq_prob,
                                              axis=1,
                                              keepdims=True)

        log_seq_prob -= log_seq_prob_sum

        # Correctly initializes log_seq_prob from the length info.
        #
        # If mask is zero, then makes the current log_seq_prob zero
        # first multiplying with the mask. After that, re-initializes the
        # log_seq_prob to be "initial_log_seq_probs[1]".
        log_seq_prob = tf.math.multiply_no_nan(log_seq_prob, mask)
        log_seq_prob += tf.math.multiply_no_nan(initial_log_seq_probs[1],
                                                (1.0 - mask))

        return log_seq_prob

    mask = tf.expand_dims(
        tf.sequence_mask(
            logit_len, maxlen=max_logit_len, dtype=tf.dtypes.float32),
        axis=-1) # yapf: disable

    # We utilize the "tf.stop_gradient" API with the "tf.nest.map_structure"
    # API based on the recommendation in the following page:
    # https://www.tensorflow.org/api_docs/python/tf/scan
    gamma, accum_log_seq_prob_sum = tf.nest.map_structure(
        tf.stop_gradient,
        tf.scan(forward,
                elems=(
                    tf.transpose(log_label_prob, perm=[1, 0, 2]),
                    tf.transpose(mask, perm=[1, 0, 2])),
                initializer=(
                    initial_log_seq_probs[0],
                    tf.fill(dims=[batch_size], value=0.0)))) # yapf: disable
    gamma = tf.transpose(gamma, perm=[1, 0, 2])
    accum_log_seq_prob_sum = tf.transpose(accum_log_seq_prob_sum, [1, 0])

    zero_padded_log_label_prob = tf.pad(log_label_prob[:, 1:, :],
                                        [[0, 0], [0, 1], [0, 0]])

    delta = tf.nest.map_structure(
        tf.stop_gradient,
        tf.scan(backward,
                elems=(tf.transpose(zero_padded_log_label_prob, perm=[1, 0, 2]),
                       tf.transpose(mask, perm=[1, 0, 2])),
                reverse=True,
                initializer=initial_log_seq_probs[1])) # yapf: disable
    delta = tf.transpose(delta, perm=[1, 0, 2])

    log_seq_prob_final = _calculate_unnormalized_log_seq_prob(
        gamma, accum_log_seq_prob_sum, logit_len, label_len)

    gamma += tf.math.multiply_no_nan(log_0, (1.0 - mask))
    delta += tf.math.multiply_no_nan(log_0, (1.0 - mask))

    mask = tf.expand_dims(
        tf.sequence_mask(
            label_len, maxlen=max_label_len, dtype=tf.dtypes.float32),
        axis=1) # yapf: disable

    gamma += tf.math.multiply_no_nan(log_0, (1.0 - mask))
    delta += tf.math.multiply_no_nan(log_0, (1.0 - mask))

    return gamma, delta, log_seq_prob_final
