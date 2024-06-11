#!/usr/bin/python3
"""A module for unit-testing the Masking layer."""

# pylint: disable=invalid-name, protected-access, import-error
# pylint: disable=no-name-in-module

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

import os
import tensorflow as tf

from speech.layers import masking_layer
from signal_processing import array_test

IsSimilarArray = array_test.IsSimilarArray

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

EPS = 1e-4


class MaskingLayerUtilTest(tf.test.TestCase):
    """A class for testing methods in the masking_layer module.
    """
    def test_small_value_masking(self):
        """This method tests the _small_value_masking private method.

        In this unit test, instead of testing the whole MaskingLayer, an
        individual private method is tested separately. The entire Layer will
        be tested in the following test_masking_layer_small_value_masking_option
        unit test.
        """

        # There are twelve elements from 0.0 up to 11.0
        input_data = tf.constant([[[[0.0, 3.0, 11.0], [1.0, 4.0, 5.0]],
                                   [[10.0, 7.0, 8.0], [9.0, 6.0, 2.0]]]],
                                 dtype=tf.float32)

        # When the ratio is 0.4, it masks the smallest 40 percent of elements.
        #
        # This corresponds to elements having values of 0.0, 1.0, 2.0, 3.0.
        # (33.3 %). The element with the value of 4.0 corresponds to 41.6 %
        # (5.0 / 12.0) percentile, so, it will not be masked.
        actual_output = masking_layer._small_value_masking(input_data, [0.4])

        expected_output = tf.constant([[[[0.0, 0.0, 11.0], [0.0, 4.0, 5.0]],
                                        [[10.0, 7.0, 8.0], [9.0, 6.0, 0.0]]]],
                                      dtype=tf.float32)

        # Checks the actual output with respect to the expected output.
        self.assertAllEqual(expected_output, actual_output)

        # When the ratio is 0.0, no value will be masked.
        actual_output = masking_layer._small_value_masking(input_data, [0.0])

        expected_output = input_data

        self.assertAllEqual(expected_output, actual_output)

        # When the ratio is 1.0, entire values except the biggest one are masked.
        actual_output = masking_layer._small_value_masking(input_data, [1.0])

        expected_output = tf.constant([[[[0.0, 0.0, 11.0], [0.0, 0.0, 0.0]],
                                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]],
                                      dtype=tf.float32)

        # Checks the actual output with respect to the expected output.
        self.assertAllEqual(expected_output, actual_output)

    def test_masking_layer_small_value_masking_option(self):
        """In this test, the entire layer is tested."""
        input_data = tf.constant([[[[0.0, 3.0, 11.0], [1.0, 4.0, 5.0]],
                                   [[10.0, 7.0, 8.0], [9.0, 6.0, 2.0]]],
                                  [[[0.0, 3.0, 11.0], [1.0, 4.0, 5.0]],
                                   [[10.0, 7.0, 8.0], [9.0, 6.0, 2.0]]]],
                                 dtype=tf.float32)

        inputs = tf.keras.layers.Input(shape=input_data.shape[1:])
        masking = (masking_layer.Masking(
            masking_layer.MaskingType.SMALL_VALUE_MASKING)(inputs))
        model = tf.keras.models.Model(inputs, outputs=masking)

        actual_output = model(input_data, training=True)

        expected_output = tf.constant([[[[0.0, 0.0, 11.0], [0.0, 0.0, 0.0]],
                                        [[10.0, 7.0, 8.0], [9.0, 0.0, 0.0]]],
                                       [[[0.0, 3.0, 11.0], [0.0, 4.0, 5.0]],
                                        [[10.0, 7.0, 8.0], [9.0, 6.0, 0.0]]]],
                                      dtype=tf.float32)

        # TODO(chanw.com) Make "_scale_output" a general utility_function.
        expected_output = masking_layer._scale_output(input_data,
                                                      expected_output)

        # Checks the actual output with respect to the expected output.
        self.assertTrue(IsSimilarArray(expected_output, actual_output, EPS))

    def test_masking_layer_macro_block_dropout(self):
        """In this test, the entire layer is tested."""
        input_data = tf.constant(
            [[[[0.0, 3.0, 11.0, 8.0], [1.0, 4.0, 5.0, 7.0]],
              [[10.0, 7.0, 8.0, 5.0], [9.0, 6.0, 2.0, 4.0]]],
             [[[0.0, 3.0, 11.0, 4.0], [1.0, 4.0, 5.0, 2.0]],
              [[10.0, 7.0, 8.0, 3.0], [9.0, 6.0, 2.0, 9.0]]]],
            dtype=tf.float32)

        inputs = tf.keras.layers.Input(shape=input_data.shape[1:])
        masking = (masking_layer.Masking(
            masking_layer.MaskingType.MACRO_BLOCK_DROPOUT,
            num_blocks=2,
            keep_prob=0.5)(inputs))
        model = tf.keras.models.Model(inputs, outputs=masking)

        actual_output = model(input_data, training=True)

        expected_output = tf.constant(
            [[[[0.0, 0.0, 11.0, 8.0], [0.0, 0.0, 5.0, 7.0]],
              [[0.0, 0.0, 8.0, 5.0], [0.0, 0.0, 0.0, 0.0]]],
             [[[0.0, 0.0, 11.0, 4.0], [0.0, 0.0, 5.0, 2.0]],
              [[10.0, 7.0, 8.0, 3.0], [0.0, 0.0, 2.0, 9.0]]]],
            dtype=tf.dtypes.float32)

        # TODO(chanw.com) Make "_scale_output" a general utility_function.
        expected_output = masking_layer._scale_output(input_data,
                                                      expected_output)

        # Checks the actual output with respect to the expected output.
        self.assertTrue(IsSimilarArray(expected_output, actual_output, EPS))

    def test_masking_layer_dropout(self):
        """In this test, the entire layer is tested."""
        input_data = tf.constant(
            [[[[0.0, 3.0, 11.0, 8.0], [1.0, 4.0, 5.0, 7.0]],
              [[10.0, 7.0, 8.0, 5.0], [9.0, 6.0, 2.0, 4.0]]],
             [[[0.0, 3.0, 11.0, 4.0], [1.0, 4.0, 5.0, 2.0]],
              [[10.0, 7.0, 8.0, 3.0], [9.0, 6.0, 2.0, 9.0]]]],
            dtype=tf.float32)

        inputs = tf.keras.layers.Input(shape=input_data.shape[1:])
        masking = (masking_layer.Masking(masking_layer.MaskingType.DROPOUT,
                                         keep_prob=0.5)(inputs))
        model = tf.keras.models.Model(inputs, outputs=masking)

        actual_output = model(input_data, training=True)

        print(actual_output)

        expected_output = tf.constant(
            [[[[0.0, 3.0, 0.0, 8.0], [0.0, 4.0, 0.0, 0.0]],
              [[0.0, 7.0, 0.0, 5.0], [9.0, 6.0, 0.0, 4.0]]],
             [[[0.0, 0.0, 0.0, 0.0], [0.0, 4.0, 5.0, 0.0]],
              [[10.0, 0.0, 8.0, 3.0], [9.0, 0.0, 2.0, 0.0]]]],
            dtype=tf.dtypes.float32)

        # TODO(chanw.com) Make "_scale_output" a general utility_function.
        expected_output = masking_layer._scale_output(input_data,
                                                      expected_output)

        # Checks the actual output with respect to the expected output.
        self.assertTrue(IsSimilarArray(expected_output, actual_output, EPS))


if __name__ == "__main__":
    tf.test.main()
