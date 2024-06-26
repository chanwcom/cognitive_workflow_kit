"""Unit tests for the seq_loss_util module."""

# pylint: disable=import-error, no-member, no-name-in-module

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import os

# Third-party imports
import numpy as np
import tensorflow as tf

# Custom imports
from loss.tensorflow import seq_loss_util
from math_lib import math_util

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

LOG_0 = seq_loss_util.log_0.numpy()


def adjust_minf(inputs):
    """Allow some margins to LOG_0."""
    return tf.math.maximum(inputs, LOG_0 + 100)


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


class SeqLossUtilTest(tf.test.TestCase):

    def test_label_trans_table(self):
        """Tests the label_trans_table method."""

        # yapf: disable
        labels = tf.constant([[0, 1, 0, 2, 0, 3, 0],
                              [0, 1, 0, 2, 0, 2, 0],
                              [0, 1, 0, 1, 0, 0, 0]])
        # yapf: enable
        labels_len = tf.constant([7, 7, 5])

        actual = seq_loss_util.label_trans_table(labels, labels_len)

        expected = tf.constant(
            [[[  0.0,   0.0, LOG_0, LOG_0, LOG_0, LOG_0, LOG_0],
              [LOG_0,   0.0,   0.0,   0.0, LOG_0, LOG_0, LOG_0],
              [LOG_0, LOG_0,   0.0,   0.0, LOG_0, LOG_0, LOG_0],
              [LOG_0, LOG_0, LOG_0,   0.0,   0.0,   0.0, LOG_0],
              [LOG_0, LOG_0, LOG_0, LOG_0,   0.0,   0.0, LOG_0],
              [LOG_0, LOG_0, LOG_0, LOG_0, LOG_0,   0.0,   0.0],
              [LOG_0, LOG_0, LOG_0, LOG_0, LOG_0, LOG_0,   0.0]],
             [[  0.0,   0.0, LOG_0, LOG_0, LOG_0, LOG_0, LOG_0],
              [LOG_0,   0.0,   0.0,   0.0, LOG_0, LOG_0, LOG_0],
              [LOG_0, LOG_0,   0.0,   0.0, LOG_0, LOG_0, LOG_0],
              [LOG_0, LOG_0, LOG_0,   0.0,   0.0, LOG_0, LOG_0],
              [LOG_0, LOG_0, LOG_0, LOG_0,   0.0,   0.0, LOG_0],
              [LOG_0, LOG_0, LOG_0, LOG_0, LOG_0,   0.0,   0.0],
              [LOG_0, LOG_0, LOG_0, LOG_0, LOG_0, LOG_0,   0.0]],
             [[  0.0,  0.0,  LOG_0, LOG_0, LOG_0, LOG_0, LOG_0],
              [LOG_0,  0.0,    0.0, LOG_0, LOG_0, LOG_0, LOG_0],
              [LOG_0, LOG_0,   0.0,   0.0, LOG_0, LOG_0, LOG_0],
              [LOG_0, LOG_0, LOG_0,   0.0,  0.0,    0.0, LOG_0],
              [LOG_0, LOG_0, LOG_0, LOG_0,  0.0,    0.0, LOG_0],
              [LOG_0, LOG_0, LOG_0, LOG_0, LOG_0,   0.0,   0.0],
              [LOG_0, LOG_0, LOG_0, LOG_0, LOG_0, LOG_0,   0.0]]],
             dtype=tf.dtypes.float32) # yapf: disable

        # Checks the actual output with respect to the expected output.
        self.assertAllClose(expected, actual)

    def test_calculate_log_label_prob(self):
        batch_size = 2
        max_logit_len = 6
        num_classes = 3

        labels = tf.constant([
            [0, 1, 0, 2, 0],
            [0, 2, 0, 1, 0],
        ])

        # Sets the random seed.
        np.random.seed(0)

        # yapf: disable
        logits = np.random.normal(
            size=(batch_size, max_logit_len, num_classes)).astype(np.float32)
        # yapf: enable
        softmax_output = tf.nn.softmax(logits, axis=2)

        actual_output = seq_loss_util.calculate_log_label_prob(
            labels, softmax_output)

        expected_output = tf.constant(
            [[[-0.5375, -1.9013, -0.5375, -1.3228, -0.5375],
              [-0.5472, -0.9206, -0.5472, -3.7654, -0.5472],
              [-0.5195, -1.6209, -0.5195, -1.5728, -0.5195],
              [-1.5273, -1.7938, -1.5273, -0.4836, -1.5273],
              [-0.8135, -1.4529, -0.8135, -1.1307, -0.8135],
              [-1.5633, -0.4029, -1.5633, -2.1022, -1.5633]],
             [[-0.3135, -3.1795, -0.3135, -1.4806, -0.3135],
              [-0.9092, -2.3050, -0.9092, -0.6984, -0.9092],
              [-0.1243, -2.3483, -0.1243, -3.8484, -0.1243],
              [-2.4703, -0.8137, -2.4703, -0.7503, -2.4703],
              [-0.9565, -1.9992, -0.9565, -0.7333, -0.9565],
              [-2.6806, -0.5435, -2.6806, -1.0477, -2.6806]]])

        self.assertAllClose(expected_output, actual_output, atol=1e-4)

    def test_calculate_alpha_beta(self):
        batch_size = 3
        max_logit_len = 6
        max_label_len = 5

        # Sets the random seed.
        np.random.seed(0)

        # yapf: disable
        labels = tf.constant([[0, 1, 0, 2, 0],
                              [0, 1, 0, 1, 0],
                              [0, 1, 0, 0, 0]])
        labels_len = tf.constant([5, 5, 3])
        # yapf: enable
        actual = seq_loss_util.label_trans_table(labels, labels_len)

        label_trans_table = tf.constant(
            [[[  0.0,   0.0, LOG_0, LOG_0, LOG_0],
              [LOG_0,   0.0,   0.0,   0.0, LOG_0],
              [LOG_0, LOG_0,   0.0,   0.0, LOG_0],
              [LOG_0, LOG_0, LOG_0,   0.0,   0.0],
              [LOG_0, LOG_0, LOG_0, LOG_0,   0.0]],
             [[  0.0,   0.0, LOG_0, LOG_0, LOG_0],
              [LOG_0,   0.0,   0.0, LOG_0, LOG_0],
              [LOG_0, LOG_0,   0.0,   0.0, LOG_0],
              [LOG_0, LOG_0, LOG_0,   0.0,   0.0],
              [LOG_0, LOG_0, LOG_0, LOG_0,   0.0]],
             [[  0.0,   0.0, LOG_0, LOG_0, LOG_0],
              [LOG_0,   0.0,   0.0,   0.0, LOG_0],
              [LOG_0, LOG_0,   0.0,   0.0, LOG_0],
              [LOG_0, LOG_0, LOG_0,   0.0,   0.0],
              [LOG_0, LOG_0, LOG_0, LOG_0,   0.0]]]) # yapf: disable

        # log_pred_label_prob is the predicted prob. of each token of the label.
        #
        # In equation form it is given by log(\tilde{y}_t)_(c_l).
        # \tilde{y}_t is time-aligned model output which predictes the probabilty
        # of the token. The index is [b, t, l].
        log_pred_label_prob = (np.random.normal(size=(batch_size,
                                                      max_logit_len,
                                                      max_label_len))).astype(
                                                          np.float32)

        log_pred_label_prob = tf.nn.log_softmax(log_pred_label_prob, axis=2)
        logits_len = tf.constant([6, 5, 4])

        (alpha, beta, log_seq_prob_final) = seq_loss_util.calculate_alpha_beta(
            label_trans_table, log_pred_label_prob, labels_len, logits_len)

        # yapf: disable
        expected_alpha = tf.constant(
            [[[    0.0000,    -1.3639,  -706.5803,  -705.0305,  -705.6915],
              [   -2.1550,     0.0000,    -2.6930,    -2.6449,  -705.2604],
              [   -3.5749,     0.0000,    -0.7374,    -1.3124,    -3.7649],
              [   -4.7630,     0.0000,    -1.3359,    -0.6504,    -3.6056],
              [   -8.9860,    -1.0080,    -0.5722,    -1.8330,     0.0000],
              [  -12.0586,    -2.5800,    -1.8788,     0.0000,    -0.0005]],
             [[   -0.2232,     0.0000,  -707.0610,  -708.1540,  -706.5211],
              [   -1.8849,     0.0000,    -0.6157,  -708.0242,  -707.7512],
              [   -4.2686,    -2.6138,    -2.6094,     0.0000,  -707.8467],
              [   -4.4939,    -3.4789,    -0.9282,    -1.3302,     0.0000],
              [   -5.5958,    -2.9892,    -1.5703,    -1.8031,     0.0000],
              [ -711.8509,  -709.5349,  -707.7346,  -708.3045,  -706.8936]],
             [[   -0.3129,     0.0000,  -706.2486, -1413.7677, -1412.1516],
              [   -1.1775,    -1.5441,     0.0000,  -708.2637, -1412.1333],
              [   -1.7813,    -1.8549,     0.0000,  -709.0778,  -709.1943],
              [   -2.3928,    -1.9218,    -0.3602,  -706.8936,  -708.3117],
              [ -712.1567,  -709.1346,  -708.3028, -1416.4167, -1413.7872],
              [ -711.3849,  -709.0323,  -709.2454, -1416.8429, -1413.7872]]])

        expected_beta = tf.constant(
            [[[    0.0000,    -0.0071,    -2.2305,    -3.0075,    -4.2789],
              [   -0.1176,     0.0000,    -1.4091,    -2.1617,    -3.6177],
              [   -0.4237,     0.0000,    -0.8840,    -0.8143,    -2.3271],
              [   -1.6495,    -0.6508,    -1.1102,     0.0000,    -0.0968],
              [ -706.4589,    -0.6619,    -0.6619,     0.0000,    -0.7254],
              [ -705.7950,  -705.5073,  -705.7950,     0.0000,     0.0000]],
             [[   -3.1343,    -0.1483,     0.0000,    -1.6785,    -4.4610],
              [   -3.8763,    -3.1646,    -0.0399,     0.0000,    -2.8037],
              [ -704.5422,    -0.4430,    -0.1200,     0.0000,    -0.2807],
              [ -704.8561,  -704.8273,    -1.4269,     0.0000,    -0.2745],
              [ -705.3372,  -705.3937,  -705.5834,     0.0000,     0.0000],
              [-1413.7872, -1413.7872, -1413.7872, -1413.7872,  -706.8936]],
             [[   -0.5719,     0.0000,    -0.1838, -1411.9089, -1412.1945],
              [   -0.3688,     0.0000,    -0.4922, -1412.2404, -1412.4261],
              [   -0.8497,     0.0000,    -0.5578, -1411.7063, -1412.1611],
              [ -705.8853,     0.0000,     0.0000, -1412.4224, -1412.4574],
              [-1413.7872, -1413.7872,  -706.8936, -2120.6809, -2120.6809],
              [-1413.7872, -1413.7872,  -706.8936, -2120.6809, -2120.6809]]])
        # yapf: enable

        expected_log_seq_prob_final = tf.constant([-5.3125, -4.9338, -4.7480])
        self.assertAllClose(expected_alpha, alpha, atol=1e-4)
        self.assertAllClose(expected_beta, beta, atol=1e-4)
        self.assertAllClose(expected_log_seq_prob_final,
                            log_seq_prob_final,
                            atol=1e-4)


class CtcLossTest(tf.test.TestCase):

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
                                    [0.0, 0.0, 0.0, 0.0, 0.0]]])
        # yapf: enable
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

        with tf.GradientTape() as tape:
            tape.watch(self._logits)
            actual_loss = seq_loss_util.ctc_loss(
                blank_augmented_label["SEQ_DATA"],
                blank_augmented_label["SEQ_LEN"], self._logits,
                self._logits_len)

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


if __name__ == "__main__":
    tf.test.main()
