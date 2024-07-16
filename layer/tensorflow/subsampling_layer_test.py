"""A module for unit-testing Subsampling layers."""

# pylint: disable=invalid-name, protected-access, import-error
# pylint: disable=no-name-in-module, no-member

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import os

# Third-party imports
import numpy as np
import tensorflow as tf
from google.protobuf import any_pb2
from google.protobuf import text_format
from packaging import version

# Custom imports
from machine_learning.layers import layer_params_pb2
from machine_learning.layers import subsampling_layer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

assert version.parse(tf.__version__) > version.parse("2.0.0"), (
    "At least tensorflow 2.0 is required.")


class SubsamplingTest(tf.test.TestCase):
    """A class for unit-testing classes dervied from the Subsampling class."""

    @classmethod
    def setUpClass(cls):
        """Creates the inputs to be used in this unit test."""
        cls._BATCH_SIZE = 3
        cls._MAX_SEQ_LEN = 40
        cls._NUM_FILTERBANK_CHANNELS = 5

        cls._seq_inputs = {}
        cls._seq_inputs["SEQ_LEN"] = tf.constant([33, 40, 21])
        mask = tf.expand_dims(tf.cast(
            tf.sequence_mask(cls._seq_inputs["SEQ_LEN"]), tf.dtypes.float32),
                              axis=2)
        cls._seq_inputs["SEQ_DATA"] = tf.random.uniform(
            shape=(cls._BATCH_SIZE, cls._MAX_SEQ_LEN,
                   cls._NUM_FILTERBANK_CHANNELS)) * mask

    def test_conv1d_subsampling(self):
        """Checks the shape of the output after performing subsampling."""
        SUBSAMPLING_FACTOR = 4
        NUM_CONV_CHANNELS = 256

        # yapf: disable
        params_proto = text_format.Parse(
            f"""
            subsampling_factor: {SUBSAMPLING_FACTOR}
            class_name:  "Conv1DSubsampling"
            class_params: {{
                [type.googleapi.com/learning.Conv1DSubsamplingParams] {{
                    num_filterbank_channels: {self._NUM_FILTERBANK_CHANNELS}
                    num_conv_channels: {NUM_CONV_CHANNELS}
                }}
            }}
            """, layer_params_pb2.SubsamplingParams())
        # yapf: enable

        factory = subsampling_layer.SubsamplingFactory()
        layer = factory.create(params_proto)
        actual_output = layer(self._seq_inputs, True)

        max_seq_len = self._MAX_SEQ_LEN
        for _ in range(int(np.log2(SUBSAMPLING_FACTOR))):
            max_seq_len = tf.math.ceil(tf.math.divide(max_seq_len, 2))

        expected_shape = tf.constant((self._BATCH_SIZE, max_seq_len.numpy(),
                                      self._NUM_FILTERBANK_CHANNELS))

        self.assertAllEqual(expected_shape,
                            tf.shape(actual_output["SEQ_DATA"]))

    def test_conv1d_subsampling_dropout(self):
        """Checks the shape of the output after performing subsampling."""
        SUBSAMPLING_FACTOR = 4
        NUM_CONV_CHANNELS = 256

        # yapf: disable
        params_proto = text_format.Parse(
            f"""
            subsampling_factor: {SUBSAMPLING_FACTOR}
            class_name:  "Conv1DSubsampling"
            class_params: {{
                [type.googleapi.com/learning.Conv1DSubsamplingParams] {{
                    num_filterbank_channels: {self._NUM_FILTERBANK_CHANNELS}
                    num_conv_channels: {NUM_CONV_CHANNELS}
                    dropout_params: {{
                        seq_noise_shape: NONE
                        class_name: "BaselineDropout"
                        class_params: {{
                            [type.googleapi.com/learning.BaselineDropoutParams] {{
                                dropout_rate: 0.1
                            }}
                        }}
                    }}
                }}
            }}
            """, layer_params_pb2.SubsamplingParams())
        # yapf: enable

        factory = subsampling_layer.SubsamplingFactory()
        layer = factory.create(params_proto)
        actual_output = layer(self._seq_inputs, True)

        max_seq_len = self._MAX_SEQ_LEN
        for _ in range(int(np.log2(SUBSAMPLING_FACTOR))):
            max_seq_len = tf.math.ceil(tf.math.divide(max_seq_len, 2))

        expected_shape = tf.constant((self._BATCH_SIZE, max_seq_len.numpy(),
                                      self._NUM_FILTERBANK_CHANNELS))

        self.assertAllEqual(expected_shape,
                            tf.shape(actual_output["SEQ_DATA"]))


if __name__ == "__main__":
    tf.test.main()
