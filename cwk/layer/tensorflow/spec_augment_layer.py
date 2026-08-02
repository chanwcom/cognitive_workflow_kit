"""The implementation of the SpecAugment layer as a Keras layer.

This module implements the following two classes:
 * SpecAugment: A Keras layer for SpecAugment which is a wrapper class for the
    following SpecAugmentOperation to be used as a layer.
 * SpecAugmentOperation: An operation implementing the SpecAugment algorithm.
    Note that this SpecAugment class internally calls "speech.feature."
    spec_augmentation_tf.SpecAugmentation".
"""

# pylint: disable=no-member, import-error

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Third-party imports
import numpy as np
import tensorflow as tf
from google.protobuf import any_pb2
from packaging import version

# Custom imports
from machine_learning.layers import spec_augment_params_pb2
from math_lib.operation import operation
from speech.feature.spec_augmentation import spec_augmentation_tf
from util import proto_util

assert version.parse(tf.__version__) > version.parse("2.0.0"), (
    "At least tensorflow 2.0 is required.")


class SpecAugment(tf.keras.layers.Layer):
    """A Keras layer implementation of the SpecAugment algorithm.

        Typical usage example:

        params_proto = text_format.Parse('''
            [type.googleapi.com/learning.SpecAugmentParams] {
                num_freq_masks: 2
                max_freq_mask_size: 13
                num_time_masks: 2
                max_time_mask_size: 100
                time_mask_upper_limit: 1.0
            }
        ''', any_pb2.Any())

        layer = SpecAugment(params_proto)
        output = layer(inputs)
    """

    def __init__(self, params_proto: any_pb2.Any) -> None:
        """Initializes a SpecAugment class.

        Args:
            params_proto: A proto-message for initialization.
                The type must be an "Any" proto message type which packs
                the SpecAugmentParams message defined in
                //speech/data_augmentation/spec_augment/spec_augment_params.proto

        Returns:
            None.
        """
        super(SpecAugment, self).__init__()
        assert isinstance(
            params_proto, any_pb2.Any), (
            "\"params_proto\" should be the Any proto message.") # yapf: disable
        assert (params_proto.Is(
            spec_augment_params_pb2.SpecAugmentParams.DESCRIPTOR))

        self._operation = SpecAugmentOperation(params_proto)

        class_params = spec_augment_params_pb2.SpecAugmentParams()
        params_proto.Unpack(class_params)

        DEFAULT_DROPOUT_BYPASS_NUM_EXAMPLES = 0

        self._dropout_bypass_num_examples = tf.convert_to_tensor(
            proto_util.get_field(class_params, "dropout_bypass_num_examples",
                                 DEFAULT_DROPOUT_BYPASS_NUM_EXAMPLES),
            dtype=tf.dtypes.float64)

    def call(self, inputs: dict, training: bool=None, num_examples=0) -> dict:  # yapf: disable
        """Runs the SpecAugment layer to process the input feature.

        Args:
            inputs: A dictionary containing a batch of features.
                This dictionary at least have the following two keys:
                "SEQ_DATA": A Tensor array of a rank of three or four
                    containing features.

                    If the rank is three, we assume that the shape of the array
                    has the following shape:
                    (batch_size, number_of_feature_frames, feature_size).

                    If the rank is four, we assume that the shape of the array
                    has the following shape:
                    (batch_size, number_of_feature_frames, feature_size,
                    number_of_microphone_channels).

                "SEQ_LEN": The length of each feature.
            training: A flag indicting whether this layer is used in training.

        Returns:
            A dictionary containing mean and/or variance normalized features.
                The format is the same as the inputs with the following keys:
                "SEQ_DATA": A Tensor array of a rank of four containing
                    features.  This array has the following shape:
                    (batch_size, number_of_feature_frames, feature_size,
                    number_of_microphone_channels).
                "SEQ_LEN": The length of each feature.
                Note that there may be additional keys other than the above
                two.
        """
        assert isinstance(inputs, dict)

        assert all(key in inputs for key in ["SEQ_DATA", "SEQ_LEN"]), (
            "The \"SEQ_DATA\" and "
            "\"SEQ_LEN\" keys must be present in \"inputs\".")

        # If "training" is False, this layer is bypassed.
        if not training:
            return inputs

        def apply_specaugment(inputs):
            outputs = {}
            # Handles the case where the rank of the input is three.
            if len(inputs["SEQ_DATA"].shape) == 3:
                expanded_inputs = {}
                expanded_inputs["SEQ_DATA"] = tf.expand_dims(
                    inputs["SEQ_DATA"], axis=-1)
                expanded_inputs["SEQ_LEN"] = inputs["SEQ_LEN"]

                # TODO(chanw.com) Instead of handling this in the layer, it may be
                # better to make SpecAugment core routine handle both the rank
                # three and rank four cases.
                org_outputs = self._operation.process(expanded_inputs)
                outputs["SEQ_DATA"] = tf.squeeze(org_outputs["SEQ_DATA"],
                                                 axis=3)
                outputs["SEQ_LEN"] = org_outputs["SEQ_LEN"]

            # Handles the case where the rank of the input is four.
            elif len(inputs["SEQ_DATA"].shape) == 4:
                outputs = self._operation.process(inputs)
            else:
                raise ValueError("The input does not have a correct rank.")

            # Passes values associated with keys other than "SEQ_DATA" and
            # "SEQ_LEN" from inputs.
            for key in inputs.keys():
                if key not in ["SEQ_DATA", "SEQ_LEN"]:
                    outputs[key] = inputs[key]

            return outputs

        return tf.cond(
            tf.math.greater_equal(
                tf.cast(num_examples, dtype=tf.dtypes.float64),
                self._dropout_bypass_num_examples),
            lambda: apply_specaugment(inputs),
            lambda: tf.nest.map_structure(tf.identity, inputs))


class SpecAugmentOperation(operation.AbstractOperation):
    """An "Operation" implementing SpecAugment.

        Typical usage example:

        params_proto = text_format.Parse('''
            [type.googleapi.com/learning.SpecAugmentParams] {
                num_freq_masks: 2
                max_freq_mask_size: 13
                num_time_masks: 2
                max_time_mask_size: 100
                time_mask_upper_limit: 1.0
            }
        ''', any_pb2.Any())

        operation = spec_augment.SpecAugment(params_proto)
        outputs = operation.process(self._inputs)
    """

    def __init__(self,
                 params_proto: any_pb2.Any,
                 params_dict=None,
                 operations_dict=None) -> None:
        assert isinstance(params_proto, any_pb2.Any), (
            "The \"params_proto\" must be of the Any proto message type.")

        self._class_params_proto = None
        self._params_proto = None
        self._core_operation = None

        self.params_proto = params_proto

    @property
    def params_proto(self) -> any_pb2.Any:
        return self._params_proto

    @params_proto.setter
    def params_proto(self, params_proto: any_pb2.Any) -> None:
        assert params_proto.Is(
            spec_augment_params_pb2.SpecAugmentParams.DESCRIPTOR)
        unpacked = spec_augment_params_pb2.SpecAugmentParams()
        params_proto.Unpack(unpacked)

        config = {}
        config["num_freq_mask"] = unpacked.num_freq_masks
        config["size_freq_mask"] = unpacked.max_freq_mask_size
        config["num_time_mask"] = unpacked.num_time_masks
        config["size_time_mask"] = unpacked.max_time_mask_size

        if not unpacked.HasField("time_mask_upper_limit"):
            config["time_mask_upper_limit"] = 1.0
        else:
            config["time_mask_upper_limit"] = unpacked.time_mask_upper_limit

        self._class_params_proto = unpacked
        self._core_operation = spec_augmentation_tf.SpecAugmentationTF(config)

    def process(self, inputs: dict) -> dict:
        """Applies SpecAugment to the inputs.

        Args:
            inputs: A dictionary containing a batch of features.
                This dictionary at least have the following two keys:
                "SEQ_DATA": A Tensor array of a rank of four containing
                    features.  This array has the following shape:
                    (batch_size, number_of_feature_frames, feature_size,
                    number_of_microphone_channels).
                "SEQ_LEN": The length of each feature.

        Returns:
            A dictionary containing mean and/or variance normalized features.
                The format is the same as the inputs with the following keys:
                "SEQ_DATA": A Tensor array of a rank of four containing
                    features.  This array has the following shape:
                    (batch_size, number_of_feature_frames, feature_size,
                    number_of_microphone_channels).
                "SEQ_LEN": The length of each feature.
                Note that there may be additional keys other than the above
                two.
        """
        assert isinstance(
            inputs, dict), ("The \"inputs\" must be the dictionary type.")
        assert all(key in inputs for key in ["SEQ_DATA", "SEQ_LEN"]), (
            "The \"SEQ_DATA\", \"SEQ_LEN\" keys must be present in "
            "\"inputs\".")

        batch_examples = inputs["SEQ_DATA"]

        tf.debugging.assert_equal(4, tf.rank(batch_examples)), (
            "The input should be a tensor of rank of four with the shape of "
            "(batch_size, max_seq_length, feature_size, num_mic_channels)")
        batch_len = tf.cast(inputs["SEQ_LEN"], dtype=tf.dtypes.int32)

        tf.debugging.assert_equal(
            tf.shape(batch_examples)[0],
            tf.shape(batch_len)[0]), (
                "The batch size from the \"SEQ_DATA\" and \"SEQ_LEN\" do not "
                "match.")

        outputs = {}
        outputs["SEQ_DATA"] = self._core_operation.process(
            inputs["SEQ_DATA"], inputs["SEQ_LEN"])
        outputs["SEQ_LEN"] = inputs["SEQ_LEN"]

        # Passes values associated with keys other than "SEQ_DATA" and
        # "SEQ_LEN" from inputs.
        for key in inputs.keys():
            if key not in ["SEQ_DATA", "SEQ_LEN"]:
                outputs[key] = inputs[key]

        return outputs
