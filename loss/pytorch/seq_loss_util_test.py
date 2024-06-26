"""Unit tests for the seq_loss_util module."""

# pylint: disable=import-error, no-member, no-name-in-module

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import os
import unittest

# Third-party imports
import numpy as np
import torch

# Custom imports
from loss.pytorch import seq_loss_util

# Sets the log of minius inifinty of the float 32 type.
LOG_0 = seq_loss_util.LOG_0

#class SeqFormatConversionTest(unittest.TestCase):
#    """A class for testing methods in the seq_loss_util module."""
#    @classmethod
#    def setUpClass(cls) -> None:
#        cls._y_sparse = {}
#        cls._y_sparse["SEQ_DATA"] = torch.tensor(
#            [[1, 3, 4, 2, 5], [1, 4, 2, 5, 0]], dtype=torch.int32)
#        cls._y_sparse["SEQ_LEN"] = torch.tensor([5, 4], dtype=torch.int32)
#
#    def test_to_blank_augmented_labels_default_blank_index(self):
#        blank_index = 0
#
#        actual_output = seq_loss_util.to_blank_augmented_labels(
#            self._y_sparse, blank_index)
#
#        expected_output = {}
#        expected_output["SEQ_DATA"] = torch.tensor(
#            [[0, 2, 0, 4, 0, 5, 0, 3, 0, 6, 0],
#             [0, 2, 0, 5, 0, 3, 0, 6, 0, 0, 0]],
#            dtype=torch.int32)
#        expected_output["SEQ_LEN"] = torch.tensor([11, 9],
#                                                 dtype=torch.int32)
#        self.assertAllEqual(expected_output["SEQ_DATA"],
#                            actual_output["SEQ_DATA"])
#        self.assertAllEqual(expected_output["SEQ_LEN"],
#                            actual_output["SEQ_LEN"])
#
#    def test_to_blank_augmented_labels_default_blank_index_no_boundary_blanks(
#            self):
#        blank_index = 0
#
#        actual_output = seq_loss_util.to_blank_augmented_labels(
#            self._y_sparse, blank_index, False)
#
#        expected_output = {}
#        expected_output["SEQ_DATA"] = torch.tensor(
#            [[2, 0, 4, 0, 5, 0, 3, 0, 6], [2, 0, 5, 0, 3, 0, 6, 0, 0]],
#            dtype=torch.int32)
#        expected_output["SEQ_LEN"] = torch.tensor([9, 7], dtype=torch.int32)
#        self.assertAllEqual(expected_output["SEQ_DATA"],
#                            actual_output["SEQ_DATA"])
#        self.assertAllEqual(expected_output["SEQ_LEN"],
#                            actual_output["SEQ_LEN"])
#
#    def test_to_blank_augmented_labels_specific_blank_index(self):
#        blank_index = 6
#
#        actual_output = seq_loss_util.to_blank_augmented_labels(
#            self._y_sparse, blank_index)
#
#        expected_output = {}
#        expected_output["SEQ_DATA"] = torch.tensor(
#            [[6, 1, 6, 3, 6, 4, 6, 2, 6, 5, 6],
#             [6, 1, 6, 4, 6, 2, 6, 5, 6, 0, 0]],
#            dtype=torch.int32)
#        expected_output["SEQ_LEN"] = torch.tensor([11, 9],
#                                                 dtype=torch.int32)
#        self.assertAllEqual(expected_output["SEQ_DATA"],
#                            actual_output["SEQ_DATA"])
#        self.assertAllEqual(expected_output["SEQ_LEN"],
#                            actual_output["SEQ_LEN"])
#
#    def test_to_blank_augmented_labels_specific_blank_index_no_boundary_blanks(
#            self):
#        blank_index = 6
#
#        actual_output = seq_loss_util.to_blank_augmented_labels(
#            self._y_sparse, blank_index, False)
#
#        expected_output = {}
#        expected_output["SEQ_DATA"] = torch.tensor(
#            [[1, 6, 3, 6, 4, 6, 2, 6, 5], [1, 6, 4, 6, 2, 6, 5, 0, 0]],
#            dtype=torch.int32)
#        expected_output["SEQ_LEN"] = torch.tensor([9, 7], dtype=torch.int32)
#        self.assertAllEqual(expected_output["SEQ_DATA"],
#                            actual_output["SEQ_DATA"])
#        self.assertAllEqual(expected_output["SEQ_LEN"],
#                            actual_output["SEQ_LEN"])


class SeqLossUtilTest(unittest.TestCase):

    def test_label_trans_table(self):
        """Tests the label_trans_table method.

        In this unit test, it is assumed that "0" corresponds to the blank
        label.
        """

        # yapf: disable
        labels = torch.tensor([[0, 1, 0, 2, 0, 3, 0],
                               [0, 1, 0, 2, 0, 2, 0],
                               [0, 1, 0, 1, 0, 0, 0]])
        labels_len = torch.tensor([7, 7, 5])
        # yapf: enable

        actual = seq_loss_util.label_trans_table(labels, labels_len)

        expected = torch.tensor(
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
             dtype=torch.float32) # yapf: disable

        # Checks the actual output with respect to the expected output.
        torch.testing.assert_allclose(actual, expected)

    def test_calculate_alpha_beta(self):
        batch_size = 3
        max_logit_len = 6
        max_label_len = 5

        # yapf: disable
        labels = torch.tensor([[0, 1, 0, 2, 0],
                               [0, 1, 0, 1, 0],
                               [0, 1, 0, 0, 0]])
        labels_len = torch.tensor([5, 5, 3])
        # yapf: enable
        actual = seq_loss_util.label_trans_table(labels, labels_len)

        label_trans_table = torch.tensor(
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
        log_pred_label_prob = torch.randn(size=(batch_size, max_logit_len,
                                                max_label_len))
        log_pred_label_prob = torch.log_softmax(log_pred_label_prob, axis=2)
        logits_len = torch.tensor([6, 5, 4])

        (alpha, beta, log_seq_prob_final) = seq_loss_util.calculate_alpha_beta(
            label_trans_table, log_pred_label_prob, labels_len, logits_len)

        # yapf: disable
        expected_alpha = torch.tensor(
            [[[    0.0000,    -1.0881,  -707.7258,  -708.0544,  -708.5428],
              [    0.0000,    -0.1939,    -0.3696,    -0.9559,  -706.1531],
              [   -1.9146,    -1.0410,     0.0000,    -1.2649,    -3.1246],
              [   -3.6110,    -1.8486,    -0.7171,     0.0000,    -1.3944],
              [   -4.0316,    -0.6778,    -1.8094,     0.0000,    -0.5339],
              [   -5.3798,    -0.9123,    -0.1723,    -2.4190,     0.0000]],
             [[    0.0000,    -2.4450,  -706.6018,  -706.2614,  -709.4107],
              [    0.0000,    -0.2672,    -1.6509,  -705.3391,  -705.7469],
              [   -1.0821,    -0.8717,     0.0000,    -3.2433,  -704.1838],
              [   -0.0266,    -0.7374,    -0.7130,     0.0000,    -1.9096],
              [   -0.6087,    -0.0082,    -0.9106,    -0.8778,     0.0000],
              [ -710.2783,  -707.5999,  -710.8660,  -709.0690,  -706.8936]],
             [[   -0.5627,     0.0000,  -705.8452, -1413.8931, -1413.9279],
              [   -1.4187,     0.0000,    -0.9661,  -708.7030, -1414.7341],
              [   -3.4511,    -2.2539,    -1.4808,  -707.5606,  -706.8936],
              [   -3.9060,    -2.1698,     0.0000,  -707.5084,  -707.0101],
              [ -712.7349,  -710.0615,  -708.4570, -1414.4249, -1413.7872],
              [ -713.8657,  -710.7015,  -709.4113, -1415.2185, -1413.7872]]])
        expected_beta = torch.tensor(
            [[[   -0.9860,     0.0000,    -0.2954,    -2.1958,    -3.9582],
              [   -1.2640,     0.0000,    -0.2671,    -1.6373,    -3.2085],
              [   -1.3105,     0.0000,    -0.2677,    -0.1549,    -1.2309],
              [   -1.5800,    -0.2123,    -0.5062,     0.0000,    -0.8766],
              [ -705.0048,    -2.5524,    -2.5524,     0.0000,    -0.0811],
              [ -705.7950,  -705.5073,  -705.7950,     0.0000,     0.0000]],
             [[   -2.5423,    -0.7287,    -0.0387,     0.0000,    -0.8125],
              [   -5.5621,    -1.5363,    -1.0153,     0.0000,    -0.1636],
              [ -705.6790,    -3.8235,    -1.2994,     0.0000,    -0.2888],
              [ -705.9285,  -706.0543,    -1.4164,     0.0000,    -0.2778],
              [ -706.4599,  -706.5057,  -706.6983,     0.0000,     0.0000],
              [-1413.7872, -1413.7872, -1413.7872, -1413.7872,  -706.8936]],
             [[   -0.0057,     0.0000,    -1.3275, -1410.8773, -1411.7058],
              [   -0.6360,     0.0000,    -0.5041, -1409.7847, -1409.8673],
              [   -1.5262,     0.0000,    -0.2451, -1412.7612, -1413.2437],
              [ -705.6595,     0.0000,     0.0000, -1411.9282, -1412.2358],
              [-1413.7872, -1413.7872,  -706.8936, -2120.6809, -2120.6809],
              [-1413.7872, -1413.7872,  -706.8936, -2120.6809, -2120.6809]]])
        # yapf: enable
        expected_log_seq_prob_final = torch.tensor([-5.9462, -6.9765, -5.3152])

        torch.testing.assert_allclose(alpha, expected_alpha, atol=1e-4)
        #torch.testing.assert_allclose(beta, expected_beta, atol=1e-4)
        #torch.testing.assert_allclose(
        #    log_seq_prob_final, expected_log_seq_prob_final, atol=1e-4)


class CtcLossTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # The shape of labels is (batch_size, labels_len).
        cls._labels = torch.tensor([[1, 2, 3], [3, 2, 0]], dtype=torch.int32)
        cls._labels_len = torch.tensor([3, 2], dtype=torch.int32)

        # A tensor containing prediction values from the softmax layer.

        # The shape of y_pred is (batch_size, pred_seq_len, num_classes).
        # The number of classes is 5.
        cls._logits = torch.tensor([[[0.6, 0.1, 0.1, 0.1, 0.1],
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
        cls._logits_len = torch.tensor([7, 5])


#    def test_ctc_loss(self):
#        """Tests the ctc_loss method."""
#
#        blank_augmented_label = seq_loss_util.to_blank_augmented_labels({
#            "SEQ_DATA":
#            self._labels,
#            "SEQ_LEN":
#            self._labels_len
#        })
#
#        actual_loss = seq_loss_util.ctc_loss(blank_augmented_label["SEQ_DATA"],
#                                             blank_augmented_label["SEQ_LEN"],
#                                             self._logits, self._logits_len)
#
#        expected_loss = torch.tensor([6.115841, 4.7813644])
#
#        self.assertAllEqual(expected_loss, actual_loss)
#
#    def test_ctc_loss_gradient(self):
#        blank_augmented_label = seq_loss_util.to_blank_augmented_labels({
#            "SEQ_DATA":
#            self._labels,
#            "SEQ_LEN":
#            self._labels_len
#        })
#
#        with tf.GradientTape() as tape:
#            tape.watch(self._logits)
#            actual_loss = seq_loss_util.ctc_loss(
#                blank_augmented_label["SEQ_DATA"],
#                blank_augmented_label["SEQ_LEN"], self._logits,
#                self._logits_len)
#
#            dy_dx = tape.gradient(actual_loss, self._logits)
#
#        # yapf: disable
#        expected_output = torch.tensor(
#            [[[-0.20110, 0.17703, -0.33000,  0.17703,  0.17703],
#              [-0.11196, 0.29188, -0.39024,  0.03329,  0.17703],
#              [-0.07836, 0.18373, -0.09580, -0.17923,  0.16966],
#              [-0.08429, 0.18373,  0.09510, -0.29525,  0.10071],
#              [-0.11051, 0.17703,  0.16153, -0.14720, -0.08085],
#              [-0.04668, 0.17703,  0.17703,  0.05078, -0.35816],
#              [-0.21160, 0.20000,  0.20000,  0.20000, -0.38840]],
#             [[-0.25679, 0.17703,  0.17703,  0.17703, -0.27430],
#              [-0.04539, 0.17703,  0.17703,  0.10163, -0.41031],
#              [-0.11353, 0.17703,  0.17703, -0.11702, -0.12351],
#              [-0.12474, 0.17703,  0.29188, -0.42733,  0.08317],
#              [-0.22634, 0.17703,  0.17703, -0.30475,  0.17703],
#              [-0.00000, 0.00000,  0.00000,  0.00000,  0.00000],
#              [-0.00000, 0.00000,  0.00000,  0.00000,  0.00000]]])
#        # yapf:enable
#
#        self.assertAllClose(expected_output, dy_dx, atol=1e-05)

if __name__ == "__main__":
    unittest.main()
