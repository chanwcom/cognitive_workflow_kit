#!/usr/bin/python3
"""The implementation of the Masking layer as a Keras layer.
"""

# pylint: disable=no-member, import-error

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

import platform
from enum import Enum
import tensorflow as tf
import tensorflow_probability as tfp

from math_lib import resize
from packaging import version

# At least Python 3.4 is required because of the usage of Enum.
assert version.parse(platform.python_version()) > version.parse("3.4.0"), (
    "At least python verion 3.0 is required.")

assert version.parse(tf.__version__) > version.parse("2.0.0"), (
    "At least tensorflow 2.0 is required.")


class MaskingType(Enum):
    """The enumeration type defining the types of masking operation."""
    SMALL_VALUE_MASKING = 1
    HIGH_VALUE_MASKING = 2
    MIDDLE_VALUE_MASKING = 3
    MACRO_BLOCK_DROPOUT = 4  # Block size should be specified.
    DROPOUT = 5


class ScalingType(Enum):
    """Scaling type to be applied to the output."""
    RATE_BASED = 1
    SUM_BASED = 2


def _scale_output_sum_based(inputs, outputs):
    batch_size = tf.shape(inputs)[0]
    rank = len(tf.shape(inputs))

    # The absolute value operation is applied since we usually do not want to
    # change the sign of values in scaling. This is not an issue with RELU
    # where the output is always >= 0. However, with some non-linearities such
    # as arctan, without "tf.math.abs", the "scaling_factor" can have
    # a negative value.
    scale_factor = tf.math.abs(
        tf.math.divide_no_nan(tf.reduce_sum(inputs, axis=tf.range(1, rank)),
                              tf.reduce_sum(outputs, axis=tf.range(1, rank))))

    # Expands the dimension to have the same rank as the input.
    scale_factor = tf.reshape(scale_factor, [batch_size] + [1] * (rank - 1))

    return tf.multiply(scale_factor, outputs)


def _scale_output_rate_based(outputs, keep_prob):
    #assert keep_prob > 0, ("The keep probability must be positive.")
    return tf.math.divide(1.0, keep_prob) * outputs


def _inverse_empirical_cdf(inputs, prob):
    """Returns a value corresponding to a certain CDF value "prob".

    Refers to the following wikipage about some relevant information:
    https://en.wikipedia.org/wiki/Quantile_function.

    Args:
        inputs: An input tensor.
        prob: A floating-point tensor with the rank of 0 or 1. This tensor
            contains the cumulative probabilities.

    Returns:
        A computed inverse CDF value.
    """

    # TODO(chanw.com) This will not work for a list of "probs". Modify the
    # code to work for that case.
    #
    # TODO(chanw.com) Need to check whether there is no negative and no more
    # than 1.0.
    #
    # This processing is based on the Nearest-Neighbor approach. Refer to the
    # following website.
    # https://en.wikipedia.org/wiki/Percentile
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

    return inputs * mask


def _compute_num_block_array(num_blocks, dim):
    """Computes an integer array containing the block size.
    """
    if isinstance(num_blocks, list):
        assert len(num_blocks) == dim, (
            "The dimension of num_block does not match with \"dim\".")
    else:
        num_blocks = [num_blocks] * dim

    return num_blocks


# TODO(chanw.com) Use something similar to "noise_shape" in Keras.
# The current dimension specification is too complicated.
def _macro_block_dropout(inputs,
                         num_blocks,
                         keep_prob,
                         same_across_time_steps=False):
    """Performs macro block dropout."""
    batch_size = tf.shape(inputs)[0]

    if same_across_time_steps:
        # yapf: disable
        masking = tf.cast(
            tfp.distributions.Bernoulli(probs=keep_prob).sample(
                [batch_size, num_blocks]),
            tf.dtypes.float32)
        # yapf: enable

        # The target_shape is the input shape.
        resized_masking = resize.resize_tensor(masking, [tf.shape(inputs)[-1]])
        resized_masking = tf.reshape(resized_masking,
                                     (batch_size, 1, tf.shape(inputs)[-1]))
    else:
        num_block_array = _compute_num_block_array(num_blocks,
                                                   len(inputs.shape[1:]))
        # yapf: disable
        masking = tf.cast(
            tfp.distributions.Bernoulli(probs=keep_prob).sample(
                [batch_size] + num_block_array),
            tf.dtypes.float32)
        # yapf: enable

        # The target_shape is the input shape.
        # TODO(chanw.com) Consider using tf.image.resize.
        resized_masking = resize.resize_tensor(masking, tf.shape(inputs)[1:])

    return inputs * resized_masking


def _dropout(inputs, keep_prob, same_across_time_steps=False):
    """Performs the original random dropout."""

    if same_across_time_steps:
        sample_shape = tf.shape(inputs)[-1]
    else:
        sample_shape = tf.shape(inputs)

    masking = tf.cast(
        tfp.distributions.Bernoulli(probs=keep_prob).sample(sample_shape),
        tf.dtypes.float32)

    return inputs * masking


def _middle_value_masking(inputs, prob=None):
    # The index can take a value between from zero up to size(inputs) - 1
    # depending on the value of prob.

    # Generates two random values...
    if not prob:
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

    if not prob:
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
        * MACRO_BLOCK_DROPOUT
        * DROPOUT (ordinary dropout)
    """

    # TODO(chanw.com) Change keep_prob to dropout_rate.
    # TODO(chanw.com) Change num_blocks somewhat similar to noise_shape in
    # tf.keras.layers.Dropout.
    def __init__(self,
                 masking_type,
                 keep_prob=1.0,
                 num_blocks=5,
                 scaling_type=ScalingType.SUM_BASED,
                 same_across_time_steps=False,
                 **kwargs):
        """
        Args:
            masking_type: The four different types supported by this layer.
            keep_prob: 1.0 - dropout_rate.
            TODO(chanw.com) Change num_blocks to noise_shape.
            num_blocks: The size of the each macro block.
            scaling_type: The scaling type. It should be one of the following:
                SUM_BASED, RATE_BASED.
            same_across_time_steps: Valid only for dropout and macro_block_dropout.
                If this option is used, then it is assumed that the input has
                the following shape:
                (batch_size, time_steps, feature_size).
                The mask shape is maintained across the time-steps.

        """
        # Note that num_blocks can be also an array.
        super(Masking, self).__init__(**kwargs)

        #assert keep_prob >= 0.0 and keep_prob <= 1.0, (
        #    "The dropout rate must be in the interval [0.0, 1.0].")

        self._masking_type = masking_type
        self._num_blocks = num_blocks
        self._keep_prob = keep_prob
        self._scaling_type = scaling_type
        self._same_across_time_steps = same_across_time_steps

    def call(self, inputs, training=None):
        """Runs the Masking layer by running the appropriate masking algorithm.

        Args:
            inputs: The input Tensor.
            training: A flag representing whether it is run during the training
                phase or not.

        Returns:
            The layer output Tensor.

        """
        # TODO(chanw.com) Check the input type and the shape.
        if self._same_across_time_steps:
            tf.debugging.assert_equal(tf.rank(inputs), 3)

        # If training is False, this layer is bypassed.
        if not training:
            return inputs

        if self._masking_type == MaskingType.SMALL_VALUE_MASKING:
            outputs = _small_value_masking(inputs)
        # TODO(chanw.com) tf.nest.map_structure is very slow. Improve
        # the following routines.
        elif self._masking_type == MaskingType.MIDDLE_VALUE_MASKING:
            outputs = tf.nest.map_structure(_middle_value_masking, inputs)
        elif self._masking_type == MaskingType.HIGH_VALUE_MASKING:
            outputs = tf.nest.map_structure(_high_value_masking, inputs)
        elif self._masking_type == MaskingType.MACRO_BLOCK_DROPOUT:
            outputs = _macro_block_dropout(inputs, self._num_blocks,
                                           self._keep_prob,
                                           self._same_across_time_steps)
        elif self._masking_type == MaskingType.DROPOUT:
            outputs = _dropout(inputs, self._keep_prob,
                               self._same_across_time_steps)
        else:
            raise NotImplementedError

        # Applies scaling depending on the scaling type.
        if self._scaling_type == ScalingType.SUM_BASED:
            outputs = _scale_output_sum_based(inputs, outputs)
        elif self._scaling_type == ScalingType.RATE_BASED:
            outputs = _scale_output_rate_based(outputs, self._keep_prob)
        else:
            raise NotImplementedError

        return outputs
