#!/usr/bin/python3
"""A module for unit-testing the Masking layer."""

# pylint: disable=invalid-name, protected-access, import-error
# pylint: disable=no-name-in-module

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

import os
import tensorflow as tf

from speech.layers import distribution_normalization_layer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

_dist_norm_layer = distribution_normalization_layer.DistributionNormalization


class DistributionNormalizationLayerTest(tf.test.TestCase):
    """A class for testing methods in the masking_layer module.
    """

    def test_masking_layer_small_value_masking_option(self):
        """In this test, the entire layer is tested."""
        input_data = tf.constant([[[[0.0, 3.0, 1.0], [1.0, 2.0, 4.0]],
                                   [[-1.0, -4.0, -3.0], [0.0, -2.0, 2.0]]],
                                  [[[0.0, -3.0, -3.0], [0.0, 4.0, 5.0]],
                                   [[10.0, 7.0, 8.0], [9.0, 6.0, 2.0]]]],
                                 dtype=tf.float32)

        inputs = tf.keras.layers.Input(shape=input_data.shape[1:])
        dist_norm = _dist_norm_layer()(inputs)
        model = tf.keras.models.Model(inputs=inputs, outputs=dist_norm)

        actual_output = model(input_data)

        print(input_data)
        print(actual_output)

        expected_output = tf.constant([[[[0.0, 0.0, 11.0], [0.0, 0.0, 0.0]],
                                        [[10.0, 7.0, 8.0], [9.0, 0.0, 0.0]]],
                                       [[[0.0, 3.0, 11.0], [0.0, 4.0, 5.0]],
                                        [[10.0, 7.0, 8.0], [9.0, 6.0, 0.0]]]],
                                      dtype=tf.float32)

        self.assertAllEqual(expected_output, actual_output)


if __name__ == "__main__":
    tf.test.main()
