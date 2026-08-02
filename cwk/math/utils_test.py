#!/usr/bin/python3
"""Unit tests for utils."""

# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import unittest
import numpy as np
import tensorflow as tf

from speech.trainer.tf_based_end_to_end_trainer.tf_trainer.util \
    import utils
from speech.trainer.tf_based_end_to_end_trainer.tf_trainer.util.utils \
    import expand_dims_from_front, squeeze_dims_from_front


class UtilsTest(unittest.TestCase):
    """ Test methods for util functions
    """

    def __init__(self, *args, **kwargs):
        super(UtilsTest, self).__init__(*args, **kwargs)

    def test_expand_dims_from_front(self):
        np.testing.assert_array_equal(
            [10], expand_dims_from_front([[10]], 1))
        np.testing.assert_array_equal(
            [1, -10], expand_dims_from_front([[-10]], 2))
        np.testing.assert_array_equal(
            [1, 10, 5, 3], expand_dims_from_front([[10, 5, 3]], 4))
        np.testing.assert_array_equal(
            [1, 1, -10, 5, 3], expand_dims_from_front([[1, -10, 5, 3]], 5))
        np.testing.assert_array_equal(
            [1, ]*10, expand_dims_from_front([[]], 10))

        with np.testing.assert_raises(AssertionError):
            # Need nested input, [[10]] instead of [10]
            expand_dims_from_front([10], 1)

        with np.testing.assert_raises(AssertionError):
            # n_dim must be bigger than len(input_shape)
            expand_dims_from_front([5, 10], 1)

    def test_squeeze_dims_from_front(self):
        def _get_tensor(shape):
            return tf.Variable(np.zeros(shape))

        def _get_shape_after_run(shape, target_shape):
            tensor = _get_tensor(shape)
            return squeeze_dims_from_front(tensor, target_shape).shape

        np.testing.assert_array_equal(
            [10], _get_shape_after_run([1, 10], [10]))
        np.testing.assert_array_equal(
            [1, 10], _get_shape_after_run([1, 10], [1, 10]))
        np.testing.assert_array_equal(
            [1, 10], _get_shape_after_run([1, 1, 1, 10], [1, 10]))

        with np.testing.assert_raises(AssertionError):
            # Invalid target shape
            _get_shape_after_run([1, 10], [2])

        with np.testing.assert_raises(AssertionError):
            # Invalid target shape
            _get_shape_after_run([1, 10], [1, 1, 10])

    def test_compute_mask_indices(self):
        mask = utils.compute_mask_indices((3, 30),
                                          mask_prob=0.3,
                                          mask_length=5,
                                          padding_mask=np.array([2, 4, 6]))

        # assert masking is valid
        np.testing.assert_equal(np.all(mask[0, -2:] == False), True)
        np.testing.assert_equal(np.all(mask[1, -4:] == False), True)
        np.testing.assert_equal(np.all(mask[2, -6:] == False), True)


if __name__ == "__main__":
    unittest.main()
