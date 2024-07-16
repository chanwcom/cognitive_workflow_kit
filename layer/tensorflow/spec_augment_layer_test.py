"""A module for unit-testing the SpecAugment layer."""

# pylint: disable=invalid-name, protected-access, import-error
# pylint: disable=no-name-in-module, no-member

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

# Standard imports
import copy
import os

# Third-party imports
import numpy as np
import tensorflow as tf
from google.protobuf import any_pb2
from google.protobuf import text_format
from packaging import version

# Custom imports
from machine_learning.layers import spec_augment_layer
from speech.trainer.util import tfdeterminism

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

assert version.parse(tf.__version__) > version.parse("2.0.0"), (
    "At least tensorflow 2.0 is required.")


class TestModel(tf.keras.Model):

    def __init__(self, params_proto) -> None:
        super(TestModel, self).__init__()

        # TODO(chanw.com) Check the type.

        self._num_examples = tf.Variable(0,
                                         dtype=tf.dtypes.int64,
                                         trainable=False)

        self._layer = spec_augment_layer.SpecAugment(params_proto)

    def call(self, inputs_dict: dict, training: bool = True) -> dict:
        """Returns the model output given a batch of inputs.

        Args:
            inputs: A dictionary containing an acoustic feature sequence.
                The keys are as follows:
                "SEQ_DATA": The acoustic feature sequence whose rank is three
                    or four. If the rank is three, the shape is as follows:
                    (batch_size, feature_len, feature_size).
                    or four. If the rank is four, the shape is as follows:
                    (batch_size, feature_len, feature_size,
                    num_microphone_channels).
                "SEQ_LEN": The length of acoust feature sequences. The shape is
                    (batch_size, 1).
            training: A flag to indicate whether it is called for training.

        Returns:
            A dictionary containing the model output.
                The keys are as follows:
                "SEQ_DATA": A model output sequence whose shape is the same as
                    the input shape.
                "SEQ_LEN": The length of model ouputs. The shape is
                    (batch_size, 1).
        """

        # yapf: disable
        return self._layer(
            inputs_dict, training=training, num_examples=self._num_examples)
        # yapf: enable

    def model_callback(self, num_examples: tf.Variable) -> bool:
        self._num_examples = copy.copy(num_examples)


class SpecAugmentTest(tf.test.TestCase):
    """A class for unit-testing SpecAugment class."""

    @classmethod
    def setUpClass(cls):
        """Creates the inputs to be used in this unit test."""

        cls._inputs = {}
        # yapf: disable
        cls._inputs["SEQ_DATA"] = tf.constant(
            [[[[ 0], [ 1], [ 2], [ 3], [ 4]],
              [[ 5], [ 6], [ 7], [ 8], [ 9]],
              [[10], [11], [12], [13], [14]],
              [[15], [16], [17], [18], [19]]],
             [[[20], [21], [22], [23], [24]],
              [[25], [26], [27], [28], [29]],
              [[30], [31], [32], [33], [34]],
              [[ 0], [ 0], [ 0], [ 0], [ 0]]]])
        cls._inputs["SEQ_LEN"] = tf.constant([4, 3])
        # yapf: enable

    def setUp(self):
        """Sets the random seed before each method is executed."""
        tf.random.set_seed(0)

    def test_single_freq_time_mask_case_rank_three_case(self):
        """In this test, the entire layer is tested."""
        # yapf: disable
        params_proto = text_format.Parse("""
            [type.googleapi.com/learning.SpecAugmentParams] {
                num_freq_masks: 1
                max_freq_mask_size: 3
                num_time_masks: 1
                max_time_mask_size: 3
            }
        """, any_pb2.Any())
        # yapf: enable

        rank_three_inputs = {}
        rank_three_inputs["SEQ_DATA"] = tf.squeeze(self._inputs["SEQ_DATA"],
                                                   axis=3)
        rank_three_inputs["SEQ_LEN"] = self._inputs["SEQ_LEN"]

        model = TestModel(params_proto)
        actual_output = model(rank_three_inputs, training=True)

        expected_output = {}
        # yapf: disable
        expected_output["SEQ_DATA"] = tf.constant(
            [[[ 0,  0,  0,  0,  0],
              [ 5,  6,  0,  8,  9],
              [10, 11,  0, 13, 14],
              [15, 16,  0, 18, 19]],
             [[ 0,  0,  0,  0,  0],
              [ 0,  0,  0,  0,  0],
              [30,  31, 32, 33, 34],
              [ 0,  0,  0,  0,  0]]])
        # yapf: enable
        expected_output["SEQ_LEN"] = tf.constant([4, 3])

        # Checks the actual output with respect to the expected output.
        self.assertAllClose(expected_output, actual_output)

    def test_single_freq_time_mask_case_rank_three_bypass_case(self):
        """In this test, the entire layer is tested."""
        # yapf: disable
        params_proto = text_format.Parse("""
            [type.googleapi.com/learning.SpecAugmentParams] {
                num_freq_masks: 1
                max_freq_mask_size: 3
                num_time_masks: 1
                max_time_mask_size: 3
                dropout_bypass_num_examples: 1000
            }
        """, any_pb2.Any())
        # yapf: enable

        rank_three_inputs = {}
        rank_three_inputs["SEQ_DATA"] = tf.squeeze(self._inputs["SEQ_DATA"],
                                                   axis=3)
        rank_three_inputs["SEQ_LEN"] = self._inputs["SEQ_LEN"]

        model = TestModel(params_proto)
        actual_output = model(rank_three_inputs, training=True)

        # Checks the actual output with respect to the expected output.
        self.assertAllClose(rank_three_inputs, actual_output)

    def test_single_freq_time_mask_case_rank_three_non_bypass_case(self):
        """In this test, the entire layer is tested."""
        # yapf: disable
        params_proto = text_format.Parse("""
            [type.googleapi.com/learning.SpecAugmentParams] {
                num_freq_masks: 1
                max_freq_mask_size: 3
                num_time_masks: 1
                max_time_mask_size: 3
                dropout_bypass_num_examples: 1000
            }
        """, any_pb2.Any())
        # yapf: enable

        rank_three_inputs = {}
        rank_three_inputs["SEQ_DATA"] = tf.squeeze(self._inputs["SEQ_DATA"],
                                                   axis=3)
        rank_three_inputs["SEQ_LEN"] = self._inputs["SEQ_LEN"]

        model = TestModel(params_proto)
        model.model_callback(2000)
        actual_output = model(rank_three_inputs, training=True)

        expected_output = {}
        # yapf: disable
        expected_output["SEQ_DATA"] = tf.constant(
            [[[ 0,  0,  0,  0,  0],
              [ 5,  6,  0,  8,  9],
              [10, 11,  0, 13, 14],
              [15, 16,  0, 18, 19]],
             [[ 0,  0,  0,  0,  0],
              [ 0,  0,  0,  0,  0],
              [30, 31, 32, 33, 34],
              [ 0,  0,  0,  0,  0]]])
        # yapf: enable
        expected_output["SEQ_LEN"] = tf.constant([4, 3])

        # Checks the actual output with respect to the expected output.
        self.assertAllClose(expected_output, actual_output)

    def test_single_freq_time_mask_case_rank_four_case(self):
        """In this test, the entire layer is tested."""
        # yapf: disable
        params_proto = text_format.Parse("""
            [type.googleapi.com/learning.SpecAugmentParams] {
                num_freq_masks: 1
                max_freq_mask_size: 3
                num_time_masks: 1
                max_time_mask_size: 3
            }
        """, any_pb2.Any())
        # yapf: enable

        model = TestModel(params_proto)
        actual_output = model(self._inputs, training=True)

        expected_output = {}
        # yapf: disable
        expected_output["SEQ_DATA"] = tf.constant(
            [[[[ 0], [ 0], [ 0], [ 0], [ 0]],
              [[ 5], [ 6], [ 0], [ 8], [ 9]],
              [[10], [11], [ 0], [13], [14]],
              [[15], [16], [ 0], [18], [19]]],
             [[[ 0], [ 0], [ 0], [ 0], [ 0]],
              [[ 0], [ 0], [ 0], [ 0], [ 0]],
              [[30], [31], [32], [33], [34]],
              [[ 0], [ 0], [ 0], [ 0], [ 0]]]])
        # yapf: enable
        expected_output["SEQ_LEN"] = tf.constant([4, 3])

        # Checks the actual output with respect to the expected output.
        self.assertAllClose(expected_output, actual_output)


class SpecAugmentOperationTest(tf.test.TestCase):
    """A class for testing the SpecAugment class."""

    @classmethod
    def setUpClass(cls):
        """Creates the input signal to be used in this unit test."""
        cls._inputs = {}
        # yapf: disable
        cls._inputs["SEQ_DATA"] = tf.constant(
            [[[[ 0], [ 1], [ 2], [ 3], [ 4]],
              [[ 5], [ 6], [ 7], [ 8], [ 9]],
              [[10], [11], [12], [13], [14]],
              [[15], [16], [17], [18], [19]]],
             [[[20], [21], [22], [23], [24]],
              [[25], [26], [27], [28], [29]],
              [[30], [31], [32], [33], [34]],
              [[ 0], [ 0], [ 0], [ 0], [ 0]]]])
        cls._inputs["SEQ_LEN"] = tf.constant([4, 3])
        # yapf: enable

    def setUp(self):
        """Sets the random seed before each method is executed."""
        tf.random.set_seed(0)

    def test_single_freq_mask_case(self):
        """Tests when there is a single frequency mask."""
        # yapf: disable
        params_proto = text_format.Parse("""
            [type.googleapi.com/learning.SpecAugmentParams] {
                num_freq_masks: 1
                max_freq_mask_size: 3
            }
        """, any_pb2.Any())
        # yapf: enable

        operation = spec_augment_layer.SpecAugmentOperation(params_proto)
        actual_output = operation.process(self._inputs)

        expected_output = {}
        # yapf: disable
        expected_output["SEQ_DATA"] = tf.constant(
            [[[[ 0], [ 1], [ 0], [ 3], [ 4]],
              [[ 5], [ 6], [ 0], [ 8], [ 9]],
              [[10], [11], [ 0], [13], [14]],
              [[15], [16], [ 0], [18], [19]]],
             [[[20], [21], [22], [23], [24]],
              [[25], [26], [27], [28], [29]],
              [[30], [31], [32], [33], [34]],
              [[ 0], [ 0], [ 0], [ 0], [ 0]]]])
        # yapf: enable
        expected_output["SEQ_LEN"] = tf.constant([4, 3])

        self.assertAllEqual(expected_output["SEQ_DATA"],
                            actual_output["SEQ_DATA"])
        self.assertAllEqual(expected_output["SEQ_LEN"],
                            actual_output["SEQ_LEN"])

    def test_single_time_mask_case(self):
        """Tests when there is a single time mask."""
        # yapf: disable
        params_proto = text_format.Parse("""
            [type.googleapi.com/learning.SpecAugmentParams] {
                num_time_masks: 1
                max_time_mask_size: 3
            }
        """, any_pb2.Any())
        # yapf: enable

        operation = spec_augment_layer.SpecAugmentOperation(params_proto)
        actual_output = operation.process(self._inputs)

        expected_output = {}
        # yapf: disable
        expected_output["SEQ_DATA"] = tf.constant(
            [[[[ 0], [ 1], [ 2], [ 3], [ 4]],
              [[ 5], [ 6], [ 7], [ 8], [ 9]],
              [[ 0], [ 0], [ 0], [ 0], [ 0]],
              [[15], [16], [17], [18], [19]]],
             [[[ 0], [ 0], [ 0], [ 0], [ 0]],
              [[25], [26], [27], [28], [29]],
              [[30], [31], [32], [33], [34]],
              [[ 0], [ 0], [ 0], [ 0], [ 0]]]])
        # yapf: enable
        expected_output["SEQ_LEN"] = tf.constant([4, 3])

        self.assertAllEqual(expected_output["SEQ_DATA"],
                            actual_output["SEQ_DATA"])
        self.assertAllEqual(expected_output["SEQ_LEN"],
                            actual_output["SEQ_LEN"])

    def test_single_freq_time_mask_case(self):
        """Tests when there is a single time mask."""
        # yapf: disable
        params_proto = text_format.Parse("""
            [type.googleapi.com/learning.SpecAugmentParams] {
                num_freq_masks: 1
                max_freq_mask_size: 3
                num_time_masks: 1
                max_time_mask_size: 3
            }
        """, any_pb2.Any())
        # yapf: enable

        operation = spec_augment_layer.SpecAugmentOperation(params_proto)
        actual_output = operation.process(self._inputs)

        expected_output = {}
        # yapf: disable
        expected_output["SEQ_DATA"] = tf.constant(
            [[[[ 0], [ 0], [ 0], [ 0], [ 0]],
              [[ 5], [ 6], [ 0], [ 8], [ 9]],
              [[10], [11], [ 0], [13], [14]],
              [[15], [16], [ 0], [18], [19]]],
             [[[ 0], [ 0], [ 0], [ 0], [ 0]],
              [[ 0], [ 0], [ 0], [ 0], [ 0]],
              [[30], [31], [32], [33], [34]],
              [[ 0], [ 0], [ 0], [ 0], [ 0]]]])
        # yapf: enable
        expected_output["SEQ_LEN"] = tf.constant([4, 3])

        self.assertAllEqual(expected_output["SEQ_DATA"],
                            actual_output["SEQ_DATA"])
        self.assertAllEqual(expected_output["SEQ_LEN"],
                            actual_output["SEQ_LEN"])

    def test_multi_freq_time_mask_case(self):
        """Tests when there is a single time mask."""
        # yapf: disable
        params_proto = text_format.Parse("""
            [type.googleapi.com/learning.SpecAugmentParams] {
                num_freq_masks: 3
                max_freq_mask_size: 2
                num_time_masks: 2
                max_time_mask_size: 2
            }
        """, any_pb2.Any())
        # yapf: enable

        operation = spec_augment_layer.SpecAugmentOperation(params_proto)
        actual_output = operation.process(self._inputs)

        expected_output = {}
        # yapf: disable
        expected_output["SEQ_DATA"] = tf.constant(
            [[[[ 0], [ 0], [ 0], [ 0], [ 0]],
              [[ 0], [ 0], [ 0], [ 0], [ 0]],
              [[10], [11], [12], [13], [ 0]],
              [[15], [16], [17], [18], [ 0]]],
             [[[ 0], [ 0], [ 0], [ 0], [ 0]],
              [[ 0], [ 0], [ 0], [ 0], [ 0]],
              [[ 0], [ 0], [ 0], [ 0], [ 0]],
              [[ 0], [ 0], [ 0], [ 0], [ 0]]]])
        # yapf: enable
        expected_output["SEQ_LEN"] = tf.constant([4, 3])

        self.assertAllEqual(expected_output["SEQ_DATA"],
                            actual_output["SEQ_DATA"])
        self.assertAllEqual(expected_output["SEQ_LEN"],
                            actual_output["SEQ_LEN"])


if __name__ == "__main__":
    tf.test.main()
