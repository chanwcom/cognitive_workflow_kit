"""Unit tests for the seq_loss_util module."""

# pylint: disable=import-error, no-member, no-name-in-module

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

# Standard imports
import os

# Third-party imports
import numpy as np
import tensorflow as tf

# Custom imports
from loss.tensorflow import seq_loss_util

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

minf = -np.inf
minf = tf.cast(tf.math.log(tf.cast(0, tf.dtypes.float64) + 1e-307),
               tf.dtypes.float32).numpy()


def adjust_minf(inputs):
    """Allow some margins to minf."""
    return tf.math.maximum(inputs, minf + 100)


class SeqFormatConversionTest(tf.test.TestCase):
    """A class for testing methods in the seq_loss_util module."""
    @classmethod
    def setUpClass(cls) -> None:
        cls._y_sparse = {}
        cls._y_sparse["SEQ_DATA"] = tf.constant(
            [[1, 3, 4, 2, 5], [1, 4, 2, 5, 0]], dtype=tf.dtypes.int32)
        cls._y_sparse["SEQ_LEN"] = tf.constant([5, 4], dtype=tf.dtypes.int32)

    def test_to_blank_augmented_labels_default_blank_index(self):
        blank_index = 0

        actual_output = seq_loss_util.to_blank_augmented_labels(
            self._y_sparse, blank_index)

        expected_output = {}
        expected_output["SEQ_DATA"] = tf.constant(
            [[0, 2, 0, 4, 0, 5, 0, 3, 0, 6, 0],
             [0, 2, 0, 5, 0, 3, 0, 6, 0, 0, 0]],
            dtype=tf.dtypes.int32)
        expected_output["SEQ_LEN"] = tf.constant([11, 9],
                                                 dtype=tf.dtypes.int32)
        self.assertAllEqual(expected_output["SEQ_DATA"],
                            actual_output["SEQ_DATA"])
        self.assertAllEqual(expected_output["SEQ_LEN"],
                            actual_output["SEQ_LEN"])

    def test_to_blank_augmented_labels_default_blank_index_no_boundary_blanks(
            self):
        blank_index = 0

        actual_output = seq_loss_util.to_blank_augmented_labels(
            self._y_sparse, blank_index, False)

        expected_output = {}
        expected_output["SEQ_DATA"] = tf.constant(
            [[2, 0, 4, 0, 5, 0, 3, 0, 6], [2, 0, 5, 0, 3, 0, 6, 0, 0]],
            dtype=tf.dtypes.int32)
        expected_output["SEQ_LEN"] = tf.constant([9, 7], dtype=tf.dtypes.int32)
        self.assertAllEqual(expected_output["SEQ_DATA"],
                            actual_output["SEQ_DATA"])
        self.assertAllEqual(expected_output["SEQ_LEN"],
                            actual_output["SEQ_LEN"])

    def test_to_blank_augmented_labels_specific_blank_index(self):
        blank_index = 6

        actual_output = seq_loss_util.to_blank_augmented_labels(
            self._y_sparse, blank_index)

        expected_output = {}
        expected_output["SEQ_DATA"] = tf.constant(
            [[6, 1, 6, 3, 6, 4, 6, 2, 6, 5, 6],
             [6, 1, 6, 4, 6, 2, 6, 5, 6, 0, 0]],
            dtype=tf.dtypes.int32)
        expected_output["SEQ_LEN"] = tf.constant([11, 9],
                                                 dtype=tf.dtypes.int32)
        self.assertAllEqual(expected_output["SEQ_DATA"],
                            actual_output["SEQ_DATA"])
        self.assertAllEqual(expected_output["SEQ_LEN"],
                            actual_output["SEQ_LEN"])

    def test_to_blank_augmented_labels_specific_blank_index_no_boundary_blanks(
            self):
        blank_index = 6

        actual_output = seq_loss_util.to_blank_augmented_labels(
            self._y_sparse, blank_index, False)

        expected_output = {}
        expected_output["SEQ_DATA"] = tf.constant(
            [[1, 6, 3, 6, 4, 6, 2, 6, 5], [1, 6, 4, 6, 2, 6, 5, 0, 0]],
            dtype=tf.dtypes.int32)
        expected_output["SEQ_LEN"] = tf.constant([9, 7], dtype=tf.dtypes.int32)
        self.assertAllEqual(expected_output["SEQ_DATA"],
                            actual_output["SEQ_DATA"])
        self.assertAllEqual(expected_output["SEQ_LEN"],
                            actual_output["SEQ_LEN"])

    def test_to_onset_augmented_labels(self):
        # The shape of y_true_sparse is (batch_size, label_seq_len).
        num_classes = 6

        actual_output = seq_loss_util.to_onset_augmented_labels(
            self._y_sparse, num_classes)

        expected_output = {}
        expected_output["SEQ_DATA"] = tf.constant(
            [[1, 7, 3, 9, 4, 10, 2, 8, 5, 11],
             [1, 7, 4, 10, 2, 8, 5, 11, 0, 0]],
            dtype=tf.dtypes.int32)
        expected_output["SEQ_LEN"] = tf.constant([10, 8],
                                                 dtype=tf.dtypes.int32)
        self.assertAllEqual(expected_output["SEQ_DATA"],
                            actual_output["SEQ_DATA"])
        self.assertAllEqual(expected_output["SEQ_LEN"],
                            actual_output["SEQ_LEN"])


class SeqLossUtilTest(tf.test.TestCase):
    """A class for testing the forced_alignment_loss module."""
    def test_label_trans_table(self):
        """Tests the label_trans_table method."""

        labels = tf.constant([[0, 1, 0, 2, 0, 3, 0], [0, 1, 0, 2, 0, 2, 0],
                              [0, 1, 0, 1, 0, 0, 0]])
        labels_len = tf.constant([7, 7, 5])

        actual = seq_loss_util.label_trans_table(labels, labels_len)

        expected = tf.constant(
            [[[ 0.0,  0.0, minf, minf, minf, minf, minf],
              [minf,  0.0,  0.0,  0.0, minf, minf, minf],
              [minf, minf,  0.0,  0.0, minf, minf, minf],
              [minf, minf, minf,  0.0,  0.0,  0.0, minf],
              [minf, minf, minf, minf,  0.0,  0.0, minf],
              [minf, minf, minf, minf, minf,  0.0,  0.0],
              [minf, minf, minf, minf, minf, minf,  0.0]],
             [[ 0.0,  0.0, minf, minf, minf, minf, minf],
              [minf,  0.0,  0.0,  0.0, minf, minf, minf],
              [minf, minf,  0.0,  0.0, minf, minf, minf],
              [minf, minf, minf,  0.0,  0.0, minf, minf],
              [minf, minf, minf, minf,  0.0,  0.0, minf],
              [minf, minf, minf, minf, minf,  0.0,  0.0],
              [minf, minf, minf, minf, minf, minf,  0.0]],
             [[ 0.0,  0.0, minf, minf, minf, minf, minf],
              [minf,  0.0,  0.0, minf, minf, minf, minf],
              [minf, minf,  0.0,  0.0, minf, minf, minf],
              [minf, minf, minf,  0.0,  0.0,  0.0, minf],
              [minf, minf, minf, minf,  0.0,  0.0, minf],
              [minf, minf, minf, minf, minf,  0.0,  0.0],
              [minf, minf, minf, minf, minf, minf,  0.0]]],
             dtype=tf.dtypes.float32) # yapf: disable

        # Checks the actual output with respect to the expected output.
        self.assertAllEqual(expected, actual)

    def test_calculate_gamma_delta(self):
        label_seq = {}
        label_seq["SEQ_DATA"] = tf.constant([[0, 1, 0, 2, 0, 2, 0],
                                             [0, 1, 0, 1, 0, 0, 0]])
        label_seq["SEQ_LEN"] = tf.constant([7, 5])

        # The shape is [batch_size, max_logit_len, max_label_len]
        log_label_prob = tf.constant(
            [[[-1., -2., -1., -2., -1., -2., -1.],
              [-1., -2., -1., -2., -1., -2., -1.],
              [-2., -1., -2., -2., -2., -2., -2.],
              [-2., -1., -2., -2., -2., -2., -2.],
              [-1., -2., -1., -2., -1., -2., -1.],
              [-2., -2., -2., -1., -2., -1., -2.],
              [-2., -2., -2., -1., -2., -1., -2.],
              [-1., -2., -1., -2., -1., -2., -1.],
              [-2., -2., -2., -1., -2., -1., -2.],
              [-1., -2., -1., -2., -1., -2., -1.]],
             [[-1., -2., -1., -2., -1., -1., -1.],
              [-1., -2., -1., -2., -1., -1., -1.],
              [-2., -1., -2., -1., -2., -2., -2.],
              [-2., -1., -2., -1., -2., -2., -2.],
              [-1., -2., -1., -2., -1., -1., -1.],
              [-2., -1., -2., -1., -2., -2., -2.],
              [-2., -1., -2., -1., -2., -2., -2.],
              [-1., -2., -1., -2., -1., -1., -1.],
              [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
              [ 0.,  0.,  0.,  0.,  0.,  0.,  0.]]]) # yapf: disable
        logit_len = tf.constant([10, 8])

        label_trans_table = seq_loss_util.label_trans_table(
            label_seq["SEQ_DATA"], label_seq["SEQ_LEN"])
        gamma, delta, log_seq_prob = seq_loss_util.calculate_gamma_delta(
            label_trans_table, log_label_prob, label_seq["SEQ_LEN"], logit_len)

        expected_gamma = tf.constant(
            [[[  0.,  -1., minf, minf, minf, minf, minf],
              [  0.,  -1.,  -1.,  -2., minf, minf, minf],
              [ -1.,   0.,  -2.,  -2.,  -3., minf, minf],
              [ -2.,   0.,  -1.,  -1.,  -3.,  -4., minf],
              [ -2.,  -1.,   0.,  -1.,  -1.,  -4.,  -4.],
              [ -3.,  -2.,  -1.,   0.,  -2.,  -1.,  -5.],
              [ -4.,  -3.,  -2.,   0.,  -1.,  -1.,  -2.],
              [ -4.,  -4.,  -2.,  -1.,   0.,  -2.,  -1.],
              [ -5.,  -5.,  -3.,  -1.,  -1.,   0.,  -2.],
              [ -5.,  -6.,  -3.,  -2.,  -1.,  -1.,   0.]],
             [[  0.,  -1., minf, minf, minf, minf, minf],
              [  0.,  -1.,  -1., minf, minf, minf, minf],
              [ -1.,   0.,  -2.,  -1., minf, minf, minf],
              [ -2.,   0.,  -1.,  -1.,  -2., minf, minf],
              [ -2.,  -1.,   0.,  -2.,  -1., minf, minf],
              [ -3.,  -1.,  -1.,   0.,  -2., minf, minf],
              [ -4.,  -1.,  -2.,   0.,  -1., minf, minf],
              [ -4.,  -2.,  -1.,  -1.,   0., minf, minf],
              [minf, minf, minf, minf, minf, minf, minf],
              [minf, minf, minf, minf, minf, minf, minf]]]) # yapf: disable
        expected_delta = tf.constant(
            [[[  0.,  -1.,  -2.,  -3.,  -3.,  -5.,  -5.],
              [  0.,   0.,  -2.,  -3.,  -3.,  -4.,  -5.],
              [  0.,   0.,  -1.,  -2.,  -2.,  -3.,  -4.],
              [ -1.,   0.,   0.,  -1.,  -1.,  -2.,  -3.],
              [ -1.,   0.,   0.,   0.,  -1.,  -1.,  -3.],
              [ -4.,   0.,   0.,   0.,  -1.,  -1.,  -2.],
              [minf,  -3.,  -3.,   0.,   0.,  -1.,  -1.],
              [minf, minf, minf,  -2.,   0.,   0.,  -1.],
              [minf, minf, minf, minf,  -1.,   0.,   0.],
              [minf, minf, minf, minf, minf,   0.,   0.]],
             [[  0.,  -1.,  -1.,  -2.,  -4., minf, minf],
              [  0.,   0.,  -1.,  -1.,  -4., minf, minf],
              [  0.,   0.,  -1.,  -1.,  -3., minf, minf],
              [ -2.,   0.,   0.,  -1.,  -2., minf, minf],
              [ -2.,  -1.,   0.,   0.,  -2., minf, minf],
              [minf,  -2.,   0.,   0.,  -1., minf, minf],
              [minf, minf,  -1.,   0.,   0., minf, minf],
              [minf, minf, minf,   0.,   0., minf, minf],
              [minf, minf, minf, minf, minf, minf, minf],
              [minf, minf, minf, minf, minf, minf, minf]]]) # yapf: disable
        expected_log_seq_prob = tf.constant([-10.0, -8.0])

        self.assertAllEqual(adjust_minf(expected_gamma), adjust_minf(gamma))
        self.assertAllEqual(adjust_minf(expected_delta), adjust_minf(delta))
        self.assertAllEqual(adjust_minf(expected_log_seq_prob),
                            adjust_minf(log_seq_prob))


class CtcLossTest(tf.test.TestCase):
    """A class for testing the forced_alignment_loss module."""
    @classmethod
    def setUpClass(cls):
        # The shape of labels is (batch_size, labels_len).
        cls._labels = tf.constant([[1, 2, 3], [3, 2, 0]],
                                  dtype=tf.dtypes.int32)
        cls._labels_len = tf.constant([3, 2], dtype=tf.dtypes.int32)

        # A tensor containing prediction values from the softmax layer.

        # The shape of y_pred is (batch_size, pred_seq_len, num_classes).
        # The number of classes is 5.
        cls._logits = tf.constant([[[0.6, 0.1, 0.1, 0.1, 0.1],
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
                                    [0.0, 0.0, 0.0, 0.0,
                                     0.0]]])  # yapf: enable
        cls._logits_len = tf.constant([7, 5])

    def test_ctc_loss(self):
        """Tests the ctc_loss method."""

        blank_augmented_label = seq_loss_util.to_blank_augmented_labels({
            "SEQ_DATA":
            self._labels,
            "SEQ_LEN":
            self._labels_len
        })

        actual_loss = seq_loss_util.ctc_loss(blank_augmented_label["SEQ_DATA"],
                                             blank_augmented_label["SEQ_LEN"],
                                             self._logits, self._logits_len)

        expected_loss = tf.constant([6.115841, 4.7813644])

        self.assertAllEqual(expected_loss, actual_loss)

    def test_ctc_loss_gradient(self):
        blank_augmented_label = seq_loss_util.to_blank_augmented_labels({
            "SEQ_DATA":
            self._labels,
            "SEQ_LEN":
            self._labels_len
        })

        smoothing_coeff = 0.0
        apply_smoothing_th = 0.0

        with tf.GradientTape() as tape:
            tape.watch(self._logits)
            actual_loss = seq_loss_util.ctc_loss(
                blank_augmented_label["SEQ_DATA"],
                blank_augmented_label["SEQ_LEN"], self._logits,
                self._logits_len, smoothing_coeff, apply_smoothing_th)

            dy_dx = tape.gradient(actual_loss, self._logits)

        # yapf: disable
        expected_output = tf.constant(
            [[[-0.20110, 0.17703, -0.33000,  0.17703,  0.17703],
              [-0.11196, 0.29188, -0.39024,  0.03329,  0.17703],
              [-0.07836, 0.18373, -0.09580, -0.17923,  0.16966],
              [-0.08429, 0.18373,  0.09510, -0.29525,  0.10071],
              [-0.11051, 0.17703,  0.16153, -0.14720, -0.08085],
              [-0.04668, 0.17703,  0.17703,  0.05078, -0.35816],
              [-0.21160, 0.20000,  0.20000,  0.20000, -0.38840]],
             [[-0.25679, 0.17703,  0.17703,  0.17703, -0.27430],
              [-0.04539, 0.17703,  0.17703,  0.10163, -0.41031],
              [-0.11353, 0.17703,  0.17703, -0.11702, -0.12351],
              [-0.12474, 0.17703,  0.29188, -0.42733,  0.08317],
              [-0.22634, 0.17703,  0.17703, -0.30475,  0.17703],
              [-0.00000, 0.00000,  0.00000,  0.00000,  0.00000],
              [-0.00000, 0.00000,  0.00000,  0.00000,  0.00000]]])
        # yapf:enable

        self.assertAllClose(expected_output, dy_dx, atol=1e-05)


class WeightedBestPathLossTest(tf.test.TestCase):
    """A class for testing the forced_alignment_loss module."""
    @classmethod
    def setUpClass(cls):
        # The shape of labels is (batch_size, labels_len).
        cls._labels = tf.constant([[1, 2, 3], [3, 2, 0]],
                                  dtype=tf.dtypes.int32)
        cls._labels_len = tf.constant([3, 2], dtype=tf.dtypes.int32)

        # A tensor containing prediction values from the softmax layer.

        # The shape of y_pred is (batch_size, pred_seq_len, num_classes).
        # The number of classes is 5.
        cls._logits = tf.constant([[[0.6, 0.1, 0.1, 0.1, 0.1],
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
                                    [0.0, 0.0, 0.0, 0.0,
                                     0.0]]])  # yapf: enable
        cls._logits_len = tf.constant([7, 5])

    def test_weighted_best_path_loss(self):
        """Tests the weighted_best_path_loss method."""

        blank_augmented_label = seq_loss_util.to_blank_augmented_labels({
            "SEQ_DATA":
            self._labels,
            "SEQ_LEN":
            self._labels_len
        })

        actual_loss = seq_loss_util.weighted_best_path_loss(
            blank_augmented_label["SEQ_DATA"],
            blank_augmented_label["SEQ_LEN"], self._logits, self._logits_len)

        expected_loss = tf.constant([10.023744, 6.657146])

        self.assertAllEqual(expected_loss, actual_loss)


class InternalMethodTest(tf.test.TestCase):
    def test_apply_smoothing(self):
        inputs = tf.constant(
            [[[0.00, 0.02, 0.98],
              [0.00, 0.00, 1.00],
              [0.80, 0.10, 0.10],
              [0.03, 0.96, 0.01]]]) # yapf: disable

        expected_output = tf.constant(
            [[[0.000, 0.100, 0.900],
              [0.050, 0.050, 0.900],
              [0.800, 0.100, 0.100],
              [0.075, 0.900, 0.025]]]) # yapf: disable

        smoothing_coeff = 0.1
        actual_output = seq_loss_util._apply_smoothing(inputs, smoothing_coeff)

        self.assertAllClose(expected_output, actual_output)

    def test_apply_smoothing_zero_trailing(self):
        inputs = tf.constant(
            [[[0.00, 0.02, 0.98],
              [0.00, 1.00, 0.00],
              [0.00, 0.00, 0.00],
              [0.00, 0.00, 0.00]]]) # yapf: disable

        expected_output = tf.constant(
            [[[0.000, 0.100, 0.900],
              [0.050, 0.900, 0.050],
              [0.000, 0.000, 0.000],
              [0.000, 0.000, 0.000]]]) # yapf: disable

        smoothing_coeff = 0.1

        actual_output = seq_loss_util._apply_smoothing(inputs, smoothing_coeff)

        self.assertAllClose(expected_output, actual_output)


if __name__ == "__main__":
    tf.test.main()
