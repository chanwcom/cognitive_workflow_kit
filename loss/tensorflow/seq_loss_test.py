"""Unit tests for the seq_loss module."""

# pylint: disable=import-error, no-member, no-name-in-module

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

# Standard imports
import os

# Third-party imports
import tensorflow as tf

# Custom imports
from loss.tensorflow import seq_loss

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class UnalignedSeqLossTest(tf.test.TestCase):
    """A class for testing the seq_loss module."""
    def test_forced_alignment(self):
        """Tests the forced_alignment method."""

        # The shape of y_true_sparse is (batch_size, label_seq_len).
        y_true_sparse = tf.constant([[1, 3, 4, 2, 5], [1, 4, 2, 5, 0]],
                                    dtype=tf.dtypes.int32)
        y_true_sparse_length = tf.constant([5, 4], dtype=tf.dtypes.int32)

        # yapf: disable
        # A NumPy array containing prediction values from the softmax layer.
        #
        # The shape of y_pred is (batch_size, pred_seq_len, num_classes).
        # Intentionally, only one class has the probability of 0.5, while other
        # classes have the probability of 0.1.
        y_pred = tf.constant([[[0.1, 0.5, 0.1, 0.1, 0.1, 0.1],
                               [0.1, 0.5, 0.1, 0.1, 0.1, 0.1],
                               [0.1, 0.1, 0.1, 0.5, 0.1, 0.1],
                               [0.1, 0.1, 0.1, 0.1, 0.5, 0.1],
                               [0.1, 0.1, 0.1, 0.1, 0.5, 0.1],
                               [0.1, 0.1, 0.1, 0.1, 0.5, 0.1],
                               [0.1, 0.1, 0.5, 0.1, 0.1, 0.1],
                               [0.1, 0.1, 0.5, 0.1, 0.1, 0.1],
                               [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
                               [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]],
                              [[0.1, 0.5, 0.1, 0.1, 0.1, 0.1],
                               [0.1, 0.1, 0.1, 0.1, 0.5, 0.1],
                               [0.1, 0.1, 0.5, 0.1, 0.1, 0.1],
                               [0.1, 0.1, 0.5, 0.1, 0.1, 0.1],
                               [0.1, 0.1, 0.5, 0.1, 0.1, 0.1],
                               [0.1, 0.1, 0.5, 0.1, 0.1, 0.1],
                               [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
                               [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        # yapf: enable
        y_pred_length = tf.constant([10, 8])

        expected_alignment = tf.constant([[1, 1, 3, 4, 4, 4, 2, 2, 5, 5],
                                          [1, 4, 2, 2, 2, 2, 5, 5, 0, 0]])

        actual_alignment = seq_loss.forced_alignment(
            tf.convert_to_tensor(y_true_sparse, dtype=tf.int32),
            tf.convert_to_tensor(y_true_sparse_length, dtype=tf.int32),
            tf.convert_to_tensor(y_pred, dtype=tf.float32),
            tf.convert_to_tensor(y_pred_length, dtype=tf.int32))[0]

        # Checks the actual output with respect to the expected output.
        self.assertAllEqual(expected_alignment, actual_alignment)

    def test_seq_loss(self):
        """Tests the UnalignedSeqLoss class."""

        y_true = {}
        # The shape of y_true_sparse is (batch_size, labels_len).
        y_true["SEQ_DATA"] = tf.constant([[1, 3, 4, 2, 5], [1, 4, 2, 5, 0]],
                                         dtype=tf.dtypes.int32)
        y_true["SEQ_LEN"] = tf.constant([5, 4])

        # yapf: disable
        # A NumPy array containing prediction values from the softmax layer.
        #
        # The shape of y_pred is (batch_size, pred_seq_len, num_classes).
        # Intentionally, only one class has the probability of 0.5, while other
        # classes have the probability of 0.1.
        y_pred = {}
        y_pred["SEQ_DATA"] = tf.constant(
            [[[0.1, 0.5, 0.1, 0.1, 0.1, 0.1],
              [0.1, 0.5, 0.1, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.5, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.1, 0.5, 0.1],
              [0.1, 0.1, 0.1, 0.1, 0.5, 0.1],
              [0.1, 0.1, 0.1, 0.1, 0.5, 0.1],
              [0.1, 0.1, 0.5, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.5, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
              [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]],
             [[0.1, 0.5, 0.1, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.1, 0.5, 0.1],
              [0.1, 0.1, 0.5, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.5, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.5, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.5, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
              [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        # yapf: enable
        y_pred["SEQ_LEN"] = tf.constant([10, 8])
        mask = tf.expand_dims(tf.cast(tf.sequence_mask(y_pred["SEQ_LEN"]),
                                      dtype=y_pred["SEQ_DATA"].dtype),
                              axis=2)

        # mask is multiplied because zero-padded portion no longer contains
        # zero values because of the logarithmic operation.
        y_pred["SEQ_DATA"] = mask * tf.math.log(
            tf.math.maximum(y_pred["SEQ_DATA"], tf.keras.backend.epsilon()))
        fa_loss = seq_loss.UnalignedSeqLoss(
            label_type=seq_loss.LabelType.NO_PROCESSING,
            alg_type=seq_loss.AlgorithmType.SINGLE_BEST_PATH_LOSS)
        actual_loss = fa_loss(y_true, y_pred)
        expected_loss = 12.4766496

        self.assertAllClose(expected_loss, actual_loss)


class UnalignedSeqLossBlankSymoblTest(tf.test.TestCase):
    """A class for testing the seq_loss module."""
    @classmethod
    def setUpClass(cls):
        # The shape of y_true_sparse is (batch_size, labels_len).
        cls._y_labels = {}
        cls._y_labels["SEQ_DATA"] = tf.constant([[1, 2, 3], [3, 1, 0]],
                                                dtype=tf.dtypes.int32)
        cls._y_labels["SEQ_LEN"] = tf.constant([3, 2], dtype=tf.dtypes.int32)

        cls._y_pred = {}
        # A tensor containing prediction values from the softmax layer.
        #
        # The shape of y_pred is (batch_size, pred_seq_len, num_classes).
        # The number of classes is 5.
        cls._y_pred["SEQ_DATA"] = tf.constant([[[0.6, 0.1, 0.1, 0.1, 0.1],
                                                [0.1, 0.6, 0.1, 0.1, 0.1],
                                                [0.3, 0.1, 0.1, 0.3, 0.1],
                                                [0.3, 0.1, 0.1, 0.3, 0.1],
                                                [0.6, 0.1, 0.1, 0.1, 0.1],
                                                [0.1, 0.1, 0.1, 0.1, 0.6],
                                                [0.2, 0.2, 0.2, 0.2, 0.2]],
                                               [[0.6, 0.1, 0.1, 0.1, 0.1],
                                                [0.1, 0.1, 0.1, 0.1, 0.6],
                                                [0.6, 0.1, 0.1, 0.1, 0.1],
                                                [0.1, 0.1, 0.6, 0.1, 0.1],
                                                [0.6, 0.1, 0.1, 0.1, 0.1],
                                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0]]])
        # yapf: enable
        cls._y_pred["SEQ_LEN"] = tf.constant([7, 5])

    def test_forced_alignment(self):
        """Tests the forced_alignment method."""

        fa_loss = seq_loss.UnalignedSeqLoss(seq_loss.LabelType.BLANK_LABEL,
                                              0)

        preprocessed_labels = fa_loss._preprocess_label(self._y_labels, 0, 5)

        actual_alignment = seq_loss.forced_alignment(
            tf.convert_to_tensor(preprocessed_labels["SEQ_DATA"],
                                 dtype=tf.int32),
            tf.convert_to_tensor(preprocessed_labels["SEQ_LEN"],
                                 dtype=tf.int32),
            tf.convert_to_tensor(self._y_pred["SEQ_DATA"], dtype=tf.float32),
            tf.convert_to_tensor(self._y_pred["SEQ_LEN"], dtype=tf.int32))[0]

        expected_alignment = tf.constant(
            [[0, 2, 0, 3, 0, 4, 0], [0, 4, 0, 2, 0, 0, 0]],
            dtype=tf.dtypes.int32)

        # Checks the actual output with respect to the expected output.
        self.assertAllEqual(expected_alignment, actual_alignment)


if __name__ == "__main__":
    tf.test.main()
