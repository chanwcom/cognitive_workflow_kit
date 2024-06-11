"""A module for forced alignment loss.

The following classes are implemented.
 * UnalignedSeqLoss
"""

# pylint: disable=no-member, invalid-name, import-error

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

# Standard imports
import enum

# Third-party imports
import numpy as np
import tensorflow as tf

# Custom imports
from loss.tensorflow import seq_loss_util


class LabelType(enum.Enum):
    # Original label is used.
    NO_PROCESSING = 1

    # Inserts blank labels.
    #
    # Blank label is inserted at the beginning of the sequence, at the end of
    # the sequence, and between labels. This scheme is basically the same as
    # that employed in CTC-training. The final length becomes 2 * N + 1.
    # The number of output classes must be C + 1 where C is the total number of
    # valid labels.
    BLANK_LABEL = 2

    # Divides each label into the on-set and the trailing label.
    #
    # Each label is divided into two portion: the on-set portion and the
    # trailing portion. The final length becomes 2 * N. The number of output
    # classes should become 2 * C where C is the total number of valid labels.
    ONSET_LABEL = 3


class AlgorithmType(enum.Enum):
    # Sequence CE loss is used.
    SEQUENCE_CE_LOSS = 1

    # Similar to the above sequence CE loss, but custom gradient is used.
    #
    # Theoretically, it should show the same result as the above
    # SEQUENCE_CE_LOSS.
    SINGLE_BEST_PATH_LOSS = 2

    # Similar to BEST_ALIGNMENT_LOSS, but weighting is applied.
    #
    # The weighting coefficient is defined to be
    #    p_best[m],c / sum_{all c} alpha_[m].
    WEIGHTED_BEST_PATH_LOSS = 3

    CTC_LOSS = 4

    CTC_BEST_PATH_LOSS = 5


# TODO(chanw.com) Check the speed and if this routine is slow, re-implement
# the code in C++.
#
# The following are some useful resources:
# * https://github.com/amaas/stanford-ctc/blob/master/ctc/ctc.py
# * https://github.com/HawkAaron/warp-transducer/blob/master/tensorflow_binding/src/warprnnt_op.cc
def _y_pred_to_y_true_label_prob(y_true_sparse, y_pred):
    return tf.math.log(tf.gather(y_pred, y_true_sparse, batch_dims=1, axis=2))


def _back_tracking_np(y_move, y_true_sparse, y_true_sparse_length,
                      y_pred_length):
    """Performs backtracking based on a NumPy array storing path movements.

    This method is called only from "back_tracking" method shown below.

    Args:
        y_move: A three-dimensional NumPy array storing movement information.
            The shape is (batch_size, label_length, temporal_length).
        y_true_sparse: The true label sequences. The shape is given by
            (batch_size, label_length).
        y_true_sparse_length: The length of y_true_sparse. The shape is given
            by (batch_size).
        y_pred_length: The length of y_pred. The shape is (batch_size).

    Returns:
        A NumPy array containing the time aligned label sequence whose shape is
        (batch_size, temporal_length).
    """
    batch_size = np.shape(y_move)[0]
    temporal_len = np.shape(y_move)[2]

    # fs stands for frame-synchronous.
    fs_indices = np.zeros((batch_size, temporal_len), dtype=np.int32)

    for b in range(0, batch_size):
        # m is the final temporal index.
        m = y_pred_length[b] - 1
        # l is the last label index.
        l = y_true_sparse_length[b] - 1

        # Traverses back through the temporal indices.
        for m in range(m, -1, -1):
            # Sets the frame-synchronous label index at the time index "m".
            fs_indices[b, m] = y_true_sparse[b, l]
            if y_move[b, l, m]:
                l -= 1

    return fs_indices


#@tf.function
def forced_alignment(y_true_sparse, y_true_sparse_length, y_pred,
                     y_pred_length):
    """Performs forced alignment.

    Args:
        y_true_sparse: The ground-truth label sequences. The shape is given by
            (batch_size, label_length).
        y_true_sparse_length: The length of y_true_sparse that has the shape of
            (batch_size).
        y_pred: The predicted "logit value". The shape is given by
            (batch_size, sequence_length, num_classes).
        y_pred_length: The length of y_pred that has the shape of (batch_size).

    Note that zero values are assumed to be masked-values.

    Returns:
        A tuple containing the alignment sequence and the losses.
            alignment_sequence: Time aligned label sequence whose shape is
                (batch_size, y_pred_length).
            loss: The final losses associated with each example. The shape is
                (batch_size).
    """
    # Note that we use the following symbol in the code comments:
    #
    # B: The batch size.
    # L: The length of a label sequence.
    # M: The length of a prediction sequence.
    # C: The number of classes in the classifier.

    # Note that the shape of y_true_sparse is (B, L).
    tf.debugging.assert_equal(tf.rank(y_true_sparse), 2)
    # Note that the shape of y_pred is (B, M, C).
    tf.debugging.assert_equal(tf.rank(y_pred), 3)
    # Checks the batch size.
    tf.debugging.assert_equal(tf.shape(y_true_sparse)[0], tf.shape(y_pred)[0])

    batch_size = tf.shape(y_true_sparse)[0]
    label_len = tf.shape(y_true_sparse)[1]
    pred_batch_len = tf.shape(y_pred)[1]

    # Obtains the true label probability at each time index.
    #
    # The shape of "y_true_label_prob" is (B, M, L).
    y_true_label_prob = _y_pred_to_y_true_label_prob(
        y_true_sparse, tf.nn.softmax(y_pred, axis=-1))

    # Transposes y_true_label_prob so that it has the shape of (M, B, L).
    #
    # This transposition has been done for easier calculation of "log_seq_prob"
    # in the later part of this method.
    # Refer to the last line inside the for loop in this method which is shown
    # below for your convenience.
    #    log_seq_prob = (tf.maximum(log_seq_prob_prev_label, log_seq_prob) +
    #                    tf.gather(y_true_label_prob, m))
    y_true_label_prob = tf.transpose(y_true_label_prob, [1, 0, 2])

    # Initializes the log sequence probability.
    #
    # The shape of this tensor is (B, L).
    log_seq_prob = tf.fill(dims=(batch_size, label_len), value=-np.inf)

    final_loss = tf.fill(dims=[batch_size], value=-np.inf)

    # Calculates log sequence probability at the time index 0 (m=0).
    #
    # The shape of log_seq_prob is (B, L). Initially, only the first label is
    # the allowed location in the alignment path. Thus, for all the other
    # labels, we assign -infinity probabilities.

    # Updates the log_seq_prob for M=0 and L=0.
    log_seq_prob = tf.transpose(
        tf.tensor_scatter_nd_update(tf.transpose(log_seq_prob), [[0]],
                                    [y_true_label_prob[0, :, 0]]))

    # A tensor containing movements in the (frame, label) plane.
    #
    # The shape of y_move is (B, L, M).
    y_move = tf.zeros(shape=(batch_size, label_len, pred_batch_len),
                      dtype=tf.dtypes.int32)

    def while_body(m, log_seq_prob, y_move, final_loss):
        # Creates a tensor containing flags about label transition.
        #
        # TODO(chanw.com) The following explanation is not good.
        #
        # Note that when the sequence probability at the previous label is
        # larger  than the current label, the path transition from the previous
        # label to the current label needs to occur. Thus, we compare the log
        # sequence probability from the previous label and the current label.
        # The shape of label_transition_flag is (B, L).
        log_seq_prob_prev_label = tf.pad(log_seq_prob[:, :-1],
                                         tf.constant([[0, 0], [1, 0]]),
                                         constant_values=-np.inf)
        label_transition_flag = tf.math.greater(log_seq_prob_prev_label,
                                                log_seq_prob)

        # Finds the indices where label_transition_flag is True.
        ids = tf.cast(tf.where(tf.math.equal(label_transition_flag, True)),
                      dtype=tf.dtypes.int32)

        # The time index "m" is appended.
        time_index = m * tf.ones((tf.shape(ids)[0], 1), dtype=tf.dtypes.int32)
        ids = tf.concat((ids, time_index), axis=1)

        # Updates the y_move using the index.
        y_move = tf.tensor_scatter_nd_update(
            y_move, ids, tf.ones((tf.shape(ids)[0]), dtype=tf.dtypes.int32))

        # Calculates the log sequence probability at the frame index m.
        #
        # Note that for the first label, there is no previous label.
        # Thus, log_seq_prob[:, 0] is used for the first label.
        log_seq_prob = (tf.maximum(log_seq_prob_prev_label, log_seq_prob) +
                        tf.gather(y_true_label_prob, m))

        ids = tf.where(tf.math.equal(m, y_pred_length - 1))
        if tf.size(ids) > 0:
            updates = tf.squeeze(tf.gather(
                tf.gather(log_seq_prob, y_true_sparse_length - 1,
                          batch_dims=1), ids),
                                 axis=1)
            final_loss = tf.tensor_scatter_nd_update(final_loss, ids, updates)
        m += 1

        return m, log_seq_prob, y_move, final_loss

    m = tf.constant(1, dtype=tf.dtypes.int32)
    y_move, final_loss = tf.while_loop(
        lambda m, _1, _2, _3: tf.less(m, pred_batch_len), while_body,
        [m, log_seq_prob, y_move, final_loss])[2:4]

    # To ignore an invalid loss case.
    #
    # If y_true_converted["SEQ_LEN"] < y_pred["SEQ_LEN"], then the loss is not
    # valid.
    # TODO(chanw.com) Understand why tf.nn.ctc.loss still returns a value
    # even in this invalid case.
    mask = tf.dtypes.cast(tf.math.greater_equal(
        tf.cast(y_pred_length, dtype=tf.dtypes.int64),
        tf.cast(y_true_sparse_length, dtype=tf.dtypes.int64)),
                          dtype=tf.float32)

    # The final_loss can be infinity when there is no valid path. This is
    # possible when label_length > pred_len. tf.math.multiply_no_nan avoids
    # that problem.
    loss = -tf.math.multiply_no_nan(final_loss, mask)

    # Performs back tracking to obtain final time-synchronous label sequences.
    aligned_path = tf.numpy_function(
        _back_tracking_np,
        [y_move, y_true_sparse, y_true_sparse_length, y_pred_length],
        tf.dtypes.int32)

    # The shape is lost because of the "tf.numpy_function" used above.
    # It may be a bug of Tensorflow, but it is needed to prevent the
    # following error message.
    #
    #  ValueError: Cannot take the length of shape with unknown rank.
    aligned_path.set_shape([y_true_sparse.shape[0], y_pred.shape[1]])

    return (aligned_path, loss)


@tf.custom_gradient
def best_alignment_loss(y_true_sparse, y_true_sparse_len, y_pred, y_pred_len):
    """Calculates the loss along the best pass.

    Args:
        y_true_sparse: The ground-truth label sequences. The shape is given by
            (batch_size, label_length).
        y_true_sparse_length: The length of y_true_sparse that has the shape of
            (batch_size).
        y_pred: The predicted "logit value". The shape is given by
            (batch_size, sequence_length, num_classes).
        y_pred_length: The length of y_pred that has the shape of (batch_size).

    Note that zero values are assumed to be masked-values.

    Returns:
        A tuple containing (loss, grad)
    """
    # Note that we use the following symbol in the code comments:
    #
    # B: The batch size.
    # L: The length of a label sequence.
    # M: The length of a prediction sequence.
    # C: The number of classes in the classifier.

    # Note that the shape of y_true_sparse is (B, L).
    tf.debugging.assert_equal(tf.rank(y_true_sparse), 2)
    # Note that the shape of y_pred is (B, M, C).
    tf.debugging.assert_equal(tf.rank(y_pred), 3)
    # Checks the batch size.
    tf.debugging.assert_equal(tf.shape(y_true_sparse)[0], tf.shape(y_pred)[0])

    best_path, loss = forced_alignment(y_true_sparse, y_true_sparse_len,
                                       y_pred, y_pred_len)
    best_path = tf.reshape(best_path, [-1])

    def grad(upstream):
        batch_size = tf.shape(y_true_sparse)[0]
        label_len = tf.shape(y_true_sparse)[1]
        max_pred_len = tf.shape(y_pred)[1]

        # Time index.
        m_indices = tf.tile(tf.range(0, max_pred_len), [batch_size])
        # Batch index.
        b_indices = tf.repeat(
            tf.range(0, batch_size),
            max_pred_len * tf.ones(batch_size, dtype=tf.dtypes.int32))

        # Creates the indices.
        indices = tf.stack((b_indices, m_indices, best_path), axis=1)
        updates = tf.fill([tf.shape(indices)[0]], 1.0)
        alignment_pos = tf.scatter_nd(indices, updates, shape=tf.shape(y_pred))

        # https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        gradient = tf.nn.softmax(y_pred, axis=2) - alignment_pos

        # To ignore an invalid loss case.
        #
        # If y_true_converted["SEQ_LEN"] < y_pred["SEQ_LEN"], then the loss is not
        # valid.
        # TODO(chanw.com) Understand why tf.nn.ctc.loss still returns a value
        # even in this invalid case.
        mask = tf.dtypes.cast(tf.math.greater_equal(
            tf.cast(y_pred_len, dtype=tf.dtypes.int64),
            tf.cast(y_true_sparse_len, dtype=tf.dtypes.int64)),
                              dtype=tf.float32)
        mask = tf.reshape(mask, (-1, 1, 1))
        gradient = tf.math.multiply(gradient, mask)

        mask = tf.expand_dims(
            tf.cast(tf.sequence_mask(y_pred_len, maxlen=max_pred_len),
                dtype=gradient.dtype), axis=2) # yapf: disable

        gradient = tf.math.multiply(gradient, mask)
        gradient = tf.math.multiply(gradient, tf.reshape(upstream, (-1, 1, 1)))

        return [None, None, gradient, None]

    return loss, grad


def sequence_ce_loss(y_true, y_true_len, y_pred, y_pred_len):
    """Calculates the loss along the best pass.

    Args:
        y_true: The ground-truth label sequences. The shape is given by
            (batch_size, label_length).
        y_true_len: The length of y_true_sparse that has the shape of
            (batch_size).
        y_pred: The predicted "logit value". The shape is given by
            (batch_size, sequence_length, num_classes).
        y_pred_len: The length of y_pred that has the shape of (batch_size).

    Note that zero values are assumed to be masked-values.

    Returns:
        The loss value.
    """
    # The shape of "aligned_y_true_sparse" is (batch_size, y_pred_length).
    aligned_y_true, loss = forced_alignment(
        tf.cast(y_true, dtype=tf.int32), tf.cast(y_true_len, dtype=tf.int32),
        tf.cast(y_pred, dtype=tf.float32), tf.cast(y_pred_len, dtype=tf.int32))

    # The loss calculated using the above "forced_alignment" should show the
    # same results as below:
    #
    #    num_classes = tf.shape(y_pred)[2]
    #    mask = tf.cast(tf.sequence_mask(y_pred_len), dtype=tf.dtypes.float32)
    #    # yapf: disable
    #    # Calculates the sequence cross-entropy.
    #
    #    import tensorflow_addons as tfa
    #
    #    loss = tf.math.reduce_sum(
    #        tfa.seq2seq.sequence_loss(
    #            y_pred,
    #            tf.one_hot(aligned_y_true, depth=num_classes),
    #            weights=mask,
    #            average_across_timesteps=False,
    #            average_across_batch=False,
    #            sum_over_timesteps=False,
    #            sum_over_batch=False), axis=1)
    #    # yapf: enable

    return loss


class UnalignedSeqLoss(tf.keras.losses.Loss):
    """A class implementing various unaligned sequence losses.

     Instead of the CTC loss, in this case, we apply forced alignment to the
    logit output to obtain label boundaries. Sequence loss is calculated
    between the logit output (y_pred) and the aligned label.

        Typical usage example:
            fa_loss = forced_alignment_loss.UnalignedSeqLoss()
            actual_loss = fa_loss(y_true, y_pred)
    """
    def __init__(
            self,
            label_type: LabelType=LabelType.BLANK_LABEL,
            blank_index=0,
            alg_type: AlgorithmType=AlgorithmType.SINGLE_BEST_PATH_LOSS,
            smoothing_coeff: float=0.0,
            boundary_blanks: bool=True,
            apply_smoothing_th: float=0.0
            ) -> None: # yapf: disable
        """Creates the UnalignedSeqLoss object.

        Returns:
            None.
        """
        super(UnalignedSeqLoss, self).__init__()
        self._label_type = label_type
        self._blank_index = blank_index
        self._alg_type = alg_type
        self._smoothing_coeff = smoothing_coeff
        self._boundary_blanks = boundary_blanks
        self._apply_smoothing_th = apply_smoothing_th

    def call(self, y_true, y_pred):
        """Calculates the forced alignment loss from "y_true" and "y_pred".

        Instead of the CTC loss, in this case, we apply forced alignment to the
        logit output to obtain label boundaries. Sequence loss is calculated
        between the logit output (y_pred_logits) and the aligned label.

        Args:
            y_true: A dict containing the ground truth value.
                SEQ_DATA: A sparse tensor containing the ground truth values.
                    The shape is (batch_size, sequence_length).
                SEQ_LEN: A tensor of rank one containing the length of each
                    ground truth sequence. The shape is (batch_size).
            y_pred: A dict containing predicted "logit" values by the model.
                SEQ_DATA: A tensor of rank three containing the "logit" values
                    from the model. The shape is (batch_size, sequence_length,
                    num_classes).
                SEQ_LEN: A tensor of rank one containing the length of each
                    logits. The shape is (batch_size, 1).

        Returns:
            A tensor value containing the calculated sequence loss.
        """
        assert isinstance(y_true, dict)
        assert isinstance(y_pred, dict)
        assert {"SEQ_DATA", "SEQ_LEN"} <= y_true.keys()
        assert {"SEQ_DATA", "SEQ_LEN"} <= y_pred.keys()

        # Note that we use the following symbol in the code comments:
        #
        # B: The batch size.
        # L: The length of a label sequence.
        # M: The length of a prediction sequence.
        # C: The number of classes in the classifier.

        # Note that the shape of y_true_sparse is (B, L).
        tf.debugging.assert_equal(tf.rank(y_true["SEQ_DATA"]), 2)
        # Note that the shape of y_pred_logits is (B, M, C).
        tf.debugging.assert_equal(tf.rank(y_pred["SEQ_DATA"]), 3)
        # Checks the batch size.
        tf.debugging.assert_equal(
            tf.shape(y_true["SEQ_DATA"])[0],
            tf.shape(y_pred["SEQ_DATA"])[0])

        num_classes = tf.shape(y_pred["SEQ_DATA"])[2]

        # Checks whether the largest sparse index is a valid number.
        tf.debugging.assert_less_equal(tf.math.reduce_max(y_true["SEQ_DATA"]),
                                       num_classes - 1)

        y_true_converted = self._preprocess_label(y_true, self._blank_index,
                                                  num_classes,
                                                  self._boundary_blanks)

        if self._alg_type == AlgorithmType.SEQUENCE_CE_LOSS:
            loss = sequence_ce_loss(
                tf.cast(y_true_converted["SEQ_DATA"], dtype=tf.int32),
                tf.cast(y_true_converted["SEQ_LEN"], dtype=tf.int32),
                tf.cast(y_pred["SEQ_DATA"], dtype=tf.float32),
                tf.cast(y_pred["SEQ_LEN"], dtype=tf.int32))
        elif self._alg_type == AlgorithmType.SINGLE_BEST_PATH_LOSS:
            loss = seq_loss_util.best_path_loss(
                tf.cast(y_true_converted["SEQ_DATA"], dtype=tf.int32),
                tf.cast(y_true_converted["SEQ_LEN"], dtype=tf.int32),
                tf.cast(y_pred["SEQ_DATA"], dtype=tf.float32),
                tf.cast(y_pred["SEQ_LEN"], dtype=tf.int32),
                tf.cast(self._smoothing_coeff, dtype=tf.float32))
        elif self._alg_type == AlgorithmType.WEIGHTED_BEST_PATH_LOSS:
            loss = seq_loss_util.weighted_best_path_loss(
                tf.cast(y_true_converted["SEQ_DATA"], dtype=tf.int32),
                tf.cast(y_true_converted["SEQ_LEN"], dtype=tf.int32),
                tf.cast(y_pred["SEQ_DATA"], dtype=tf.float32),
                tf.cast(y_pred["SEQ_LEN"], dtype=tf.int32),
                tf.cast(self._smoothing_coeff, dtype=tf.float32))
        elif self._alg_type == AlgorithmType.CTC_LOSS:
            loss = seq_loss_util.ctc_loss(
                tf.cast(y_true_converted["SEQ_DATA"], dtype=tf.int32),
                tf.cast(y_true_converted["SEQ_LEN"], dtype=tf.int32),
                tf.cast(y_pred["SEQ_DATA"], dtype=tf.float32),
                tf.cast(y_pred["SEQ_LEN"], dtype=tf.int32),
                tf.cast(self._smoothing_coeff, dtype=tf.float32),
                tf.cast(self._apply_smoothing_th, dtype=tf.float32))
        elif self._alg_type == AlgorithmType.CTC_BEST_PATH_LOSS:
            loss = seq_loss_util.ctc_best_path_loss(
                tf.cast(y_true_converted["SEQ_DATA"], dtype=tf.int32),
                tf.cast(y_true_converted["SEQ_LEN"], dtype=tf.int32),
                tf.cast(y_pred["SEQ_DATA"], dtype=tf.float32),
                tf.cast(y_pred["SEQ_LEN"], dtype=tf.int32),
                tf.cast(self._smoothing_coeff, dtype=tf.float32))
        else:
            raise ValueError("Unsupported algorithm type is given.")

        # To ignore an invalid loss case.
        #
        # If y_true_converted["SEQ_LEN"] < y_pred["SEQ_LEN"], then the loss is not
        # valid.
        # TODO(chanw.com) Understand why tf.nn.ctc.loss still returns a value
        # even in this invalid case.
        mask = tf.dtypes.cast(tf.math.greater_equal(
            tf.cast(y_pred["SEQ_LEN"], dtype=tf.dtypes.int64),
            tf.cast(y_true_converted["SEQ_LEN"], dtype=tf.dtypes.int64)),
                              dtype=tf.float32)

        return tf.math.reduce_sum(loss * mask)

    def _preprocess_label(self,
                          inputs: dict,
                          blank_index: int,
                          num_classes: int,
                          boundary_blanks: bool=True) -> dict: # yapf: disable
        assert isinstance(inputs, dict)
        assert {"SEQ_DATA", "SEQ_LEN"} <= inputs.keys()

        if self._label_type == LabelType.NO_PROCESSING:
            outputs = inputs
        elif self._label_type == LabelType.BLANK_LABEL:
            outputs = seq_loss_util.to_blank_augmented_labels(
                inputs, blank_index, boundary_blanks)
        elif self._label_type == LabelType.ONSET_LABEL:
            outputs = seq_loss_util.to_onset_augmented_labels(
                inputs, int(num_classes / 2))
        else:
            raise ValueError("Unsupported label type is given.")

        return outputs
