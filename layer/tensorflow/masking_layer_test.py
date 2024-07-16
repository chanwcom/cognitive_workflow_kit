#!/usr/bin/python3
"""A module for unit-testing the Masking layer."""

# pylint: disable=invalid-name, protected-access, import-error
# pylint: disable=no-name-in-module, no-member

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

# Standard imports
import os

# Third-party imports
import tensorflow as tf
import tensorflow_probability as tfp
from packaging import version

# Custom imports
from machine_learning.layers import masking_layer
from speech.trainer.util import tfdeterminism

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

assert version.parse(tf.__version__) > version.parse("2.0.0"), (
    "At least tensorflow 2.0 is required.")

# It has been observed that the generated random values are different when
# version 0.7.0 of "tensorflow_probability" is used.
assert version.parse(tfp.__version__) >= version.parse("0.11.0"), (
    "At least tensorflow_probability 0.11.0 is required.")


class MaskingLayerUtilTest(tf.test.TestCase):
    """A class for testing methods in the masking_layer module."""

    @classmethod
    def setUpClass(cls):
        """Sets the tf determinism for reproducible results."""
        tfdeterminism.set_global_determinism()

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
        self.assertAllClose(expected_output, actual_output)

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

        inputs = tf.keras.Input(shape=input_data.shape[1:])
        masking = (masking_layer.Masking(
            masking_layer.MaskingType.SMALL_VALUE_MASKING)(inputs))
        model = tf.keras.models.Model(inputs, outputs=masking)

        actual_output = model(input_data, training=True)

        expected_output = tf.constant([[[[0.0, 0.0, 11.0], [0.0, 0.0, 0.0]],
                                        [[10.0, 7.0, 8.0], [9.0, 0.0, 0.0]]],
                                       [[[0.0, 3.0, 11.0], [0.0, 4.0, 5.0]],
                                        [[10.0, 7.0, 8.0], [9.0, 6.0, 0.0]]]],
                                      dtype=tf.float32)

        # TODO(chanw.com) Make "_scale_output_sum_based" a general
        # utility function.
        expected_output = masking_layer._scale_output_sum_based(
            input_data, expected_output)

        # Checks the actual output with respect to the expected output.
        self.assertAllClose(expected_output, actual_output)

    def test_masking_layer_macro_block_dropout(self):
        """In this test, the entire layer is tested."""
        input_data = tf.constant(
            [[[[0.0, 3.0, 11.0, 8.0], [1.0, 4.0, 5.0, 7.0]],
              [[10.0, 7.0, 8.0, 5.0], [9.0, 6.0, 2.0, 4.0]]],
             [[[0.0, 3.0, 11.0, 4.0], [1.0, 4.0, 5.0, 2.0]],
              [[10.0, 7.0, 8.0, 3.0], [9.0, 6.0, 2.0, 9.0]]]],
            dtype=tf.float32)

        inputs = tf.keras.Input(shape=input_data.shape[1:])
        masking = (masking_layer.Masking(
            masking_layer.MaskingType.MACRO_BLOCK_DROPOUT,
            num_blocks=2,
            keep_prob=0.5)(inputs))
        model = tf.keras.models.Model(inputs, outputs=masking)

        actual_output = model(input_data, training=True)

        expected_output = tf.constant(
            [[[[0.0, 3.0, 11.0, 8.0], [0.0, 0.0, 0.0, 0.0]],
              [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
             [[[0.0, 3.0, 11.0, 4.0], [1.0, 4.0, 0.0, 0.0]],
              [[0.0, 0.0, 8.0, 3.0], [0.0, 0.0, 0.0, 0.0]]]],
            dtype=tf.dtypes.float32)

        # TODO(chanw.com) Make "_scale_output_sum_based" a general
        # utility function.
        expected_output = masking_layer._scale_output_sum_based(
            input_data, expected_output)

        # Checks the actual output with respect to the expected output.
        self.assertAllClose(expected_output, actual_output)

    def test_masking_layer_macro_block_dropout_avoid_all_zeros(self):
        """Tests whether the avoid_all_zeros option works as expected.

        We intentionally choose a very small keep_prob of 0.001 to see
        whether all values are zeros or not.
        """
        input_data = tf.constant(
            [[[[0.0, 3.0, 11.0, 8.0], [1.0, 4.0, 5.0, 7.0]],
              [[10.0, 7.0, 8.0, 5.0], [9.0, 6.0, 2.0, 4.0]]],
             [[[0.0, 3.0, 11.0, 4.0], [1.0, 4.0, 5.0, 2.0]],
              [[10.0, 7.0, 8.0, 3.0], [9.0, 6.0, 2.0, 9.0]]]],
            dtype=tf.float32)

        inputs = tf.keras.Input(shape=input_data.shape[1:])

        # Case 1, tests when "avoid_all_zeros" is False.
        #
        # In this case, since keep_prob is so low, all the elements are zero.
        masking = (masking_layer.Masking(
            masking_layer.MaskingType.MACRO_BLOCK_DROPOUT,
            num_blocks=2,
            keep_prob=0.001,
            avoid_all_zeros=False)(inputs))
        model = tf.keras.models.Model(inputs, outputs=masking)

        actual_output = model(input_data, training=True)

        expected_output = tf.zeros(tf.shape(input_data),
                                   dtype=tf.dtypes.float32)

        self.assertAllClose(expected_output, actual_output)

        # Case 2, tests when "avoid_all_zeros" is True.
        masking = (masking_layer.Masking(
            masking_layer.MaskingType.MACRO_BLOCK_DROPOUT,
            num_blocks=2,
            keep_prob=0.01,
            avoid_all_zeros=True)(inputs))
        model = tf.keras.models.Model(inputs, outputs=masking)

        actual_output = model(input_data, training=True)

        expected_output = tf.constant(
            [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
              [[0.0, 0.0, 0.0, 0.0], [9.0, 6.0, 0.0, 0.0]]],
             [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
              [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
            dtype=tf.float32)

        # TODO(chanw.com) Make "_scale_output_sum_based" a general
        # utility function.
        expected_output = masking_layer._scale_output_sum_based(
            input_data, expected_output)

        # Checks the actual output with respect to the expected output.
        self.assertAllClose(expected_output, actual_output)

    def test_masking_layer_macro_block_dropout_same_across_time_steps(self):
        """In this unit test, the testing configuration is as follows:

        mask_type: MACRO_BLOCK_DROPOUT
        scale_type: ScalingType.SUM_BASED
        same_across_time_steps: True
        """
        input_data = tf.constant(
            [[[0.0, 3.0, 11.0, 8.0], [1.0, 4.0, 5.0, 7.0]],
             [[10.0, 7.0, 8.0, 5.0], [9.0, 6.0, 2.0, 4.0]],
             [[0.0, 3.0, 11.0, 4.0], [1.0, 4.0, 5.0, 2.0]],
             [[10.0, 7.0, 8.0, 3.0], [9.0, 6.0, 2.0, 9.0]]],
            dtype=tf.float32)

        inputs = tf.keras.Input(shape=input_data.shape[1:])
        masking = (masking_layer.Masking(
            masking_layer.MaskingType.MACRO_BLOCK_DROPOUT,
            num_blocks=2,
            keep_prob=0.5,
            same_across_time_steps=True,
        )(inputs))
        model = tf.keras.models.Model(inputs, outputs=masking)

        actual_output = model(input_data, training=True)

        # yapf: disable
        # pylint: disable=bad-whitespace
        expected_output = tf.constant(
            [[[0.0, 3.0, 11.0, 8.0], [1.0, 4.0, 5.0, 7.0]],
             [[0.0, 0.0,  0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
             [[0.0, 0.0,  0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
             [[0.0, 0.0,  0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            dtype=tf.dtypes.float32)
        # pylint: enable=bad-whitespace
        # yapf: enable

        # TODO(chanw.com) Make "_scale_output_sum_based" a general
        # utility function.
        expected_output = masking_layer._scale_output_sum_based(
            input_data, expected_output)

        # Checks the actual output with respect to the expected output.
        self.assertAllClose(expected_output, actual_output)

    def test_masking_layer_macro_block_dropout_rate_scaling(self):
        """In this test, the entire layer is tested."""
        input_data = tf.constant(
            [[[[0.0, 3.0, 11.0, 8.0], [1.0, 4.0, 5.0, 7.0]],
              [[10.0, 7.0, 8.0, 5.0], [9.0, 6.0, 2.0, 4.0]]],
             [[[0.0, 3.0, 11.0, 4.0], [1.0, 4.0, 5.0, 2.0]],
              [[10.0, 7.0, 8.0, 3.0], [9.0, 6.0, 2.0, 9.0]]]],
            dtype=tf.float32)

        inputs = tf.keras.Input(shape=input_data.shape[1:])
        masking = (masking_layer.Masking(
            masking_layer.MaskingType.MACRO_BLOCK_DROPOUT,
            num_blocks=2,
            scaling_type=masking_layer.ScalingType.RATE_BASED,
            keep_prob=0.5)(inputs))
        model = tf.keras.models.Model(inputs, outputs=masking)

        actual_output = model(input_data, training=True)

        # Division by 0.5 is done, since the dropout rate is 0.5
        expected_output = tf.constant(
            [[[[0.0, 3.0, 11.0, 8.0], [0.0, 0.0, 0.0, 0.0]],
              [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
             [[[0.0, 3.0, 11.0, 4.0], [1.0, 4.0, 0.0, 0.0]],
              [[0.0, 0.0, 8.0, 3.0], [0.0, 0.0, 0.0, 0.0]]]],
            dtype=tf.dtypes.float32) / 0.5

        # Checks the actual output with respect to the expected output.
        self.assertAllClose(expected_output, actual_output)

    def test_masking_layer_dropout(self):
        """In this unit test, the testing configuration is as follows:

        mask_type: DROPOUT
        scale_type: ScalingType.SUM_BASED
        same_across_time_steps: True
        """
        input_data = tf.constant(
            [[[[0.0, 3.0, 11.0, 8.0], [1.0, 4.0, 5.0, 7.0]],
              [[10.0, 7.0, 8.0, 5.0], [9.0, 6.0, 2.0, 4.0]]],
             [[[0.0, 3.0, 11.0, 4.0], [1.0, 4.0, 5.0, 2.0]],
              [[10.0, 7.0, 8.0, 3.0], [9.0, 6.0, 2.0, 9.0]]]],
            dtype=tf.float32)

        inputs = tf.keras.Input(shape=input_data.shape[1:])
        masking = (masking_layer.Masking(
            masking_layer.MaskingType.DROPOUT,
            keep_prob=0.5,
            scaling_type=masking_layer.ScalingType.SUM_BASED,
            same_across_time_steps=False,
        )(inputs))
        model = tf.keras.models.Model(inputs, outputs=masking)

        actual_output = model(input_data, training=True)

        expected_output = tf.constant(
            [[[[0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
              [[10.0, 7.0, 8.0, 0.0], [0.0, 6.0, 0.0, 0.0]]],
             [[[0.0, 3.0, 11.0, 4.0], [0.0, 4.0, 5.0, 2.0]],
              [[0.0, 0.0, 0.0, 0.0], [9.0, 6.0, 2.0, 9.0]]]],
            dtype=tf.dtypes.float32)

        # TODO(chanw.com) Make "_scale_output_sum_based" a general
        # utility function.
        expected_output = masking_layer._scale_output_sum_based(
            input_data, expected_output)

        # Checks the actual output with respect to the expected output.
        self.assertAllClose(expected_output, actual_output)

    def test_masking_layer_dropout_same_mask_across_time_steps(self):
        """In this unit test, the testing configuration is as follows:

        mask_type: DROPOUT
        scale_type: ScalingType.RATE_BASED
        same_across_time_steps=True
        """
        input_data = tf.constant(
            [[[0.0, 3.0, 11.0, 8.0], [1.0, 4.0, 5.0, 7.0]],
             [[10.0, 7.0, 8.0, 5.0], [9.0, 6.0, 2.0, 4.0]],
             [[0.0, 3.0, 11.0, 4.0], [1.0, 4.0, 5.0, 2.0]],
             [[10.0, 7.0, 8.0, 3.0], [9.0, 6.0, 2.0, 9.0]]],
            dtype=tf.float32)

        inputs = tf.keras.Input(shape=input_data.shape[1:])
        masking = (masking_layer.Masking(
            masking_layer.MaskingType.DROPOUT,
            keep_prob=0.5,
            scaling_type=masking_layer.ScalingType.RATE_BASED,
            same_across_time_steps=True,
        )(inputs))
        model = tf.keras.models.Model(inputs, outputs=masking)

        actual_output = model(input_data, training=True)

        expected_output = tf.constant(
            [[[0.0, 3.0, 0.0, 0.0], [1.0, 4.0, 0.0, 0.0]],
             [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
             [[0.0, 3.0, 11.0, 0.0], [1.0, 4.0, 5.0, 0.0]],
             [[0.0, 7.0, 0.0, 0.0], [0.0, 6.0, 0.0, 0.0]]],
            dtype=tf.float32) / 0.5

        # Checks the actual output with respect to the expected output.
        self.assertAllClose(expected_output, actual_output)


if __name__ == "__main__":
    tf.test.main()
