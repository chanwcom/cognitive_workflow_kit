#!/usr/bin/python3
"""The implementation of the Masking layer as a Keras layer."""

# pylint: disable=no-member

# TODO(chanw.com) How to implement a routine which checks the version is at
# least 3.4.?
# At least Python 3.4 is required because of the usage of Enum.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

from math_lib import probability

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def _scale_outputs(inputs, outputs):
    batch_size = tf.shape(inputs)[0]
    rank = len(tf.shape(inputs))

    scale_factor = tf.math.divide_no_nan(
        tf.reduce_sum(inputs, axis=tf.range(1, rank)),
        tf.reduce_sum(outputs, axis=tf.range(1, rank)))

    scale_factor = tf.reshape(scale_factor, [batch_size] + [1] * (rank - 1))

    return tf.multiply(scale_factor, outputs)


def _inverse_gaussian_cdf(inputs):
    """Computes the inverse Gaussian Cumulative Density Function (CDF).

    Args:
        inputs: An input tensor. Each element must be in [-1.0, 1.0].

    Returns:
        An output tensor containing the inverse CDF.
    """
    # The relationship between the Cumulative Density Function (CDF) of the
    # Gaussian distribution F(x) and the error function ("erf") is given
    # by the following equation:
    #   F(x) = 0.5 + 0.5 erf(x / sqrt(2)).
    #
    # From the above relationship, we may derive that:
    #  F^{-1}(x) = \sqrt(2) erf^{-1})((x - 1/2) / (1/2)).
    #
    # Note that when g(x) = a * f(b x) + c, its inverse function is given by:
    #   g^{-1}(x) = 1 / b * f((y - c) / a).

    return np.sqrt(2.0) * tf.math.erfinv(2.0 * inputs - 1.0)
    #return np.sqrt(2.0) * tf.math.erfinv(inputs)


class DistributionNormalization(tf.keras.layers.Layer):
    """A keras layer implementation of distribution normalization.
    """

    def __init__(self, rescaling=False, **kwargs):
        super(DistributionNormalization, self).__init__(**kwargs)
        self._rescaling = rescaling

    def call(self, inputs):
        """Runs the Masking layer by running the appropriate masking algorithm.

        Args:
            inputs: The input Tensor.
            training: A flag representing whether it is run during the training
                phase or not.

        Returns:
            The layer output Tensor.

        """
        outputs = _inverse_gaussian_cdf(
            tf.math.minimum(
                tf.cast(probability.empirical_cumulative_prob(inputs),
                        tf.dtypes.float32), 0.999))

        outputs = tf.where(inputs > 0.0, outputs, inputs)

        if self._rescaling:
            outputs = _scale_outputs(inputs, outputs)

        return outputs
