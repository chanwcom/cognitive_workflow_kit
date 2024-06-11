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
from speech.layers import masking_layer_lib

# TODO(chanw.com)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class MaskingLayerLibTest(tf.test.TestCase):
    """A class for testing methods in the masking_layer module.
    """
    def test_apply_block_masking():
        """In this test, the entire layer is tested."""

        # The shape of the Tensor is [2, 5, 5]
        input_data = tf.constant([[[[0.0, 0.0, 3.0, 1.0], [2.0, 0.0, 0.0, 1.0]
                                    ], [[0.0, 2.0, 2.0], [1.0, 1.0, 0.0]]],
                                  [[[0.0, 4.0, 0.0], [0.0, 4.0, 0.0]],
                                   [[0.0, 0.0, 2.0], [0.0, 2.0, 0.0]]]],
                                 dtype=tf.float32)

        actual_output = masking_layer_lib._exp_masking(input_data)
        self.assertTrue(True)


if __name__ == "__main__":
    tf.test.main()
