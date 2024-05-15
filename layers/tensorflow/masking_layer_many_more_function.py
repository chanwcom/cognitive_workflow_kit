#!/usr/bin/python3
"""The implementation of the Masking layer as a Keras layer."""

# pylint: disable=no-member

# TODO(chanw.com) How to implement a routine which checks the version is at
# least 3.4.?
# At least Python 3.4 is required because of the usage of Enum.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

from enum import Enum
from math_lib import probability
from math_lib import resize

import tensorflow as tf
import tensorflow_probability as tfp


class MaskingType(Enum):
    """The enumeration type defining the types of masking operation."""
    SMALL_VALUE_MASKING = 1
    HIGH_VALUE_MASKING = 2
    MIDDLE_VALUE_MASKING = 3
    RANDOM_MASKING = 4
    RANDOM_BLOCK_MASKING = 5  # Block size should be specified.
    CUMULATIVE_PROB_BASED_MASKING = 6


def _inverse_empirical_cdf(inputs, prob):
    # TODO TODO(chanw.com) Update the comments.
    """Returns a value corresponding to a certain CDF value "prob".

    Refers to the following wikipage about some relevant information:
    https://en.wikipedia.org/wiki/Quantile_function.

    Args:
        inputs:
        prob: A floating-point tensor with the rank of 0 or 1. This tensor
            contains the cumulative probabilities.

    Returns:
        A computed inverse CDF value.


    """
    # This will not work properly if prob is a list of two elements.

    # TODO(chanw.com) Need to check whether there is no negative and no more
    # than 1.0.

    # The following is based on the simple nearest  TODO TODO
    #https://en.wikipedia.org/wiki/Percentile

    index = tf.maximum(
        tf.cast(
            tf.math.ceil(
                tf.multiply(tf.cast(tf.size(inputs[0]), tf.float32), prob)) -
            1, tf.int32), 0)
    batch_size = tf.shape(inputs)[0]
    reshaped_inputs = tf.reshape(inputs, (batch_size, -1))
    sorted_inputs = tf.sort(reshaped_inputs, axis=1)

    threshold = tf.gather_nd(sorted_inputs,
                             tf.stack([tf.range(batch_size), index], axis=1))

    return threshold


def _small_value_masking(inputs, probs=None):
    batch_size = tf.shape(inputs)[0]

    if probs is None:
        probs = tf.random.uniform((batch_size, ), 0.0, 1.0)

    threshold = _inverse_empirical_cdf(inputs, probs)

    # Computes the rank of inputs using the following hack.
    #
    # Note that we cannot use tf.rank.numpy() in the non-eager mode.
    rank = len(inputs.shape.as_list())

    # Expands the rank of "threshold" to match that of inputs.
    threshold = tf.reshape(threshold, [-1] + [1] * (rank - 1))
    mask = 1.0 - tf.cast(tf.math.less(inputs, threshold), dtype=inputs.dtype)

    return (inputs * mask)


def _compute_num_block_array(num_blocks, dim):
    """Computes an integer array containing  the block size.
    """
    if isinstance(num_blocks, list):
        assert (len(num_blocks) == dim)
    else:
        num_blocks = [num_blocks] * dim

    return num_blocks


def _random_block_masking(inputs, num_blocks, keep_prob):
    num_block_array = _compute_num_block_array(num_blocks,
                                               len(inputs.shape[1:]))
    batch_size = tf.shape(inputs)[0]

    masking = tf.cast(
        tfp.distributions.Bernoulli(probs=keep_prob).sample([batch_size] +
                                                            num_block_array),
        tf.dtypes.float32)

    print("++++++++++++++")
    print(masking)
    print("++++++++++++++")

    # The target_shape is the input shape.
    resized_masking = resize.resize_tensor(masking, inputs.shape[1:])

    return inputs * resized_masking


def _cumulative_prob_based_masking(inputs):
    """
    Args:
    """
    batch_size = tf.shape(inputs)[0]
    keep_prob = tf.cast(probability.empirical_cumulative_prob(inputs),
                        tf.dtypes.float32)

    masking = tf.cast(
        tfp.distributions.Bernoulli(probs=keep_prob).sample(),
        tf.dtypes.float32)

    return (tf.math.divide_no_nan(1.0, keep_prob) * inputs * masking)


def _middle_value_masking(inputs, prob=None):
    # The index can take a value between from zero up to size(inputs) - 1
    # depending on the value of prob.

    # Generates two random values...
    if prob == None:
        prob = tf.random.uniform((2, ), 0.0, 1.0)

    threshold = _inverse_empirical_cdf(inputs, prob)

    assert tf.shape(threshold) == (2, ), (
        "The dimension of the threshold is not correct.")

    # Constructs the binary mask.
    binary_mask = 1.0 - tf.cast(
        tf.logical_and(inputs <= threshold[0], inputs < threshold[1],
                       inputs.dtype))

    return tf.math.multiply(inputs, binary_mask)


def _high_value_masking(inputs, prob=None):
    # The index can take a value between from zero up to size(inputs) - 1
    # depending on the value of prob.

    if prob == None:
        prob = tf.random.uniform([], 0.0, 1.0)

    threshold = _inverse_empirical_cdf(inputs, prob)

    # Constructs the binary masks.
    binary_mask = 1.0 - tf.cast(threshold < inputs, inputs.dtype)

    return tf.math.multiply(inputs, binary_mask)


class Masking(tf.keras.layers.Layer):
    """A keras layer implementation of Masking layer.

    The following four different operations are supported.
        * SMALL_VALUE_MASKING
        * HIGH_VALUE_MASKING
        * MIDDLE_VALUE_MASKING
        * RANDOM_MASKING
    """
    def __init__(self, masking_type, keep_prob=1.0, num_blocks=4, **kwargs):
        # Note that num_blocks can be also an array.
        super(Masking, self).__init__(**kwargs)

        assert keep_prob >= 0.0 and keep_prob <= 1.0, (
            "The dropout rate must be in the interval [0.0, 1.0].")

        self._masking_type = masking_type
        self._num_blocks = num_blocks
        self._keep_prob = keep_prob

    def call(self, inputs, training=None):
        """Runs the Masking layer by running the appropriate masking algorithm.

        Args:
            inputs: The input Tensor.
            training: A flag representing whether it is run during the training
                phase or not.

        Returns:
            The layer output Tensor.

        """

        # TODO TODO(chanw.com) Let's check the input type.

        # If training is False, this layer is bypassed.
        if not training:
            return inputs

        if self._masking_type == MaskingType.SMALL_VALUE_MASKING:
            outputs = _small_value_masking(inputs)
        elif self._masking_type == MaskingType.CUMULATIVE_PROB_BASED_MASKING:
            outputs = _cumulative_prob_based_masking(inputs)
        elif self._masking_type == MaskingType.MIDDLE_VALUE_MASKING:
            return tf.nest.map_structure(_middle_value_masking, inputs)
        elif self._masking_type == MaskingType.HIGH_VALUE_MASKING:
            return tf.nest.map_structure(_high_value_masking, inputs)
        elif self._masking_type == MaskingType.RANDOM_BLOCK_MASKING:
            return _random_block_masking(inputs, self._num_blocks,
                                         self._keep_prob)
        else:
            raise NotImplementedError

            outputs = _scale_value(inputs, outputs)

        return outputs
