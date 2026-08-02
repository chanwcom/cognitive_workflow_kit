"""A module implementing the subsampling layer.

The following classes are implemented.
 * NeuralFrontend
"""

# pylint: disable=no-member, import-error

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import abc
import enum
import math

# Third-party imports
import numpy as np
import tensorflow as tf
from packaging import version

# Custom imports
from machine_learning.layers import neural_frontend_pb2
from speech.trainer.ck_trainer.util import proto_util
from speech.trainer.tf_based_end_to_end_trainer.tf_trainer.layers import normalization

assert version.parse(tf.__version__) >= version.parse("2.0.0"), (
    "At least tensorflow 2.0 is required.")


class NeuralFrontend(tf.keras.layers.Layer):
    """An abstract class for Subsampling."""

    @abc.abstractmethod
    def __init__(self,
                 params_proto: layer_params_pb2.SubsamplingParams) -> None:
        """Initializes a Subsampling object."""
        super(NeuralFrontend, self).__init__()

        assert isinstance(
            params_proto, layer_params_pb2.NeuralFrontendParams), (
                "The type of \"params_proto\" should be SubsamplingParams.")

        self._subsampling_factor = proto_util.get_field(
            params_proto, "subsampling_factor", 4)

    @property
    def call(self):
        pass


class NeuralNoiseSuppressionModule(tf.keras.layers.Layer):

    def __init__(self, params_proto):
        pass

    def call(self, inputs: dict, training: bool = None) -> dict:
        pass


class NeuralTemporalMaskingModule(tf.keras.layers.Layer):

    def __init__(self, params_proto):
        pass

    def call(self, inputs: dict, training: bool = None) -> dict:
        pass
