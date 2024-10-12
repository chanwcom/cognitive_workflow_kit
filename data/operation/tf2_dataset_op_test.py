"""A module for unit-testing the "tf2_dataset_operation" module."""

# pylint: disable=import-error, invalid-name, no-member, no-name-in-module
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

# Standard imports
import os
import unittest

# Third-party imports
import tensorflow as tf
from packaging import version

# Custom imports
from data.operation import dataset_op_params
from data.operation import tf2_dataset_op
from operation import operation

assert version.parse(tf.__version__) >= version.parse("2.0.0"), (
    "At least tensorflow 2.0 is required.")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class NTPDatasetOpTest(tf.test.TestCase):
    """A class for unit-testing the NTPDatasetOp class."""
    def setUp(self):
        self._dataset = tf.data.Dataset.from_tensor_slices([[1, 2, 3], [4, 5, 6]])

    def test_process(self):
        params =dataset_op_params.NTPDatasetOpParams()
        op = tf2_dataset_op.NTPDatasetOp(params)
        actual = []

        dataset = op.process(self._dataset)
        for data in dataset:
            actual.append(data)

        expected = [(tf.constant([1, 2]), tf.constant([2, 3])),
                    (tf.constant([4, 5]), tf.constant([5, 6]))]

        self.assertAllEqual(expected, actual)


#class BasicDatasetOperationTest(unittest.TestCase):
#    """A class for unit-testing the BasicDatasetOperation class."""
#    def setUp(self):
#        # Prepares the input and patches the batch and the cache methods.
#        self._input_dataset = tf.data.Dataset.from_tensor_slices(
#            tf.constant([0, 1, 2, 3, 4, 5]))
#
#        # Patches the batch, cache, and prefetch methods using "Mock".
#        #
#        # Note that the return value must be the original input, since the
#        # return value will be used as the dataset processed. Without this
#        # "return_value" of the original input, the mock object will returned
#        # instead.
#        tf.data.Dataset.batch = mock.Mock(return_value=self._input_dataset)
#        tf.data.Dataset.cache = mock.Mock(return_value=self._input_dataset)
#        tf.data.Dataset.prefetch = mock.Mock(return_value=self._input_dataset)
#
#    def test_batch_with_use_cache_case(self):
#        """Checks when both "batch_size" and "use_cache" are enabled."""
#        params_proto = text_format.Parse(
#            """
#            [type.googleapi.com/learning.BasicDatasetOperationParams] {
#                batch_size: 200
#                use_cache: True
#                use_prefetch: True
#            }
#        """, any_pb2.Any())
#
#        operation = tf2_dataset_operation.BasicDatasetOperation(params_proto)
#        operation.process(self._input_dataset)
#
#        self._input_dataset.batch.assert_called_once_with(200)
#        self._input_dataset.cache.assert_called_once()
#        self._input_dataset.prefetch.assert_called_once()
#
#    def test_batch_without_use_cache_case(self):
#        """Checks when "use_cache" is not enabled."""
#        params_proto = text_format.Parse(
#            """
#            [type.googleapi.com/learning.BasicDatasetOperationParams] {
#                batch_size: 200
#            }
#        """, any_pb2.Any())
#
#        operation = tf2_dataset_operation.BasicDatasetOperation(params_proto)
#        operation.process(self._input_dataset)
#
#        self._input_dataset.batch.assert_called_once_with(200)
#        self._input_dataset.cache.assert_not_called()


#class DatasetFilterOperationTest(tf.test.TestCase):
#    """A class for unit-testing the DatasetFilterOperation class."""
#    @classmethod
#    def setUpClass(cls):
#        # Prepares the input and patches the batch and the cache methods.
#        cls._input_dataset = tf.data.Dataset.from_tensor_slices(
#            tf.constant([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]))
#
#        # Patches the batch and the cache methods using "Mock".
#        #
#        # Note that the return value must be the original input, since the
#        # return value will be used as the dataset processed. Without this
#        # "return_value" of the original input, the mock object will returned
#        # instead.
#        tf.data.Dataset.batch = mock.Mock(return_value=cls._input_dataset)
#        tf.data.Dataset.cache = mock.Mock(return_value=cls._input_dataset)
#
#    def test_filter_less_equal_three(self):
#        class OperationFilter(operation.AbstractOperation):
#            """A operation which doubles the input."""
#            def __init__(self,
#                         params_proto=None,
#                         params_dict=None,
#                         operations_dict=None) -> None:
#                # pylint: disable=unused-argument
#                self._params = params_proto
#
#            @property
#            def params_proto(self):
#                """Returns the params_proto message used for initialization.
#
#                Args:
#                    None.
#
#                Returns:
#                    A proto-message containing the initialization information.
#                """
#                raise NotImplementedError
#
#            @params_proto.setter
#            def params_proto(self, params_proto):
#                """Sets the params_proto proto-message for initialization.
#
#                Args:
#                    params_proto: A proto-message for initialization.
#
#                Returns:
#                    None.
#                """
#                raise NotImplementedError
#
#            def process(self, *args, **kwargs):
#                """Returns a Boolen tensor which is True if inputs <= 3."""
#                # pylint: disable=unused-argument
#                assert len(args) == 1
#                inputs = args[0]
#
#                if isinstance(inputs, operation.Message):
#                    return inputs
#
#                return tf.math.less_equal(args[0], 3)
#
#        params_proto = text_format.Parse(
#            """
#            [type.googleapi.com/learning.DatasetFilterOperationParams] {
#                operation_params: {
#                    class_name: "OperationFilter"
#                }
#            }
#        """, any_pb2.Any())
#
#        op = tf2_dataset_operation.DatasetFilterOperation(params_proto)
#        output_dataset = op.process(self._input_dataset)
#
#        output_list = []
#        for data in output_dataset:
#            output_list.append(data.numpy())
#        actual_output = tf.constant(output_list)
#
#        expected_output = tf.constant([0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0])
#        self.assertAllClose(expected_output, actual_output)
#
#
#class DatasetWrapperOperationTest(tf.test.TestCase):
#    """A class for unit-testing the DatasetWrapperOperation class."""
#    @classmethod
#    def setUpClass(cls):
#        # Prepares the input and patches the batch and the cache methods.
#        cls._input_dataset = tf.data.Dataset.from_tensor_slices(
#            tf.constant([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
#
#        # Patches the batch and the cache methods using "Mock".
#        #
#        # Note that the return value must be the original input, since the
#        # return value will be used as the dataset processed. Without this
#        # "return_value" of the original input, the mock object will returned
#        # instead.
#        tf.data.Dataset.batch = mock.Mock(return_value=cls._input_dataset)
#        tf.data.Dataset.cache = mock.Mock(return_value=cls._input_dataset)
#
#    def test_double_operation_case(self):
#        params_proto = text_format.Parse(
#            """
#            [type.googleapi.com/learning.DatasetWrapperOperationParams] {
#                operation_params: {
#                    class_name: "OperationDouble"
#                    class_params: {
#                        [type.googleapi.com/learning.OperationDoubleParams] {
#                        }
#                    }
#                }
#            }
#        """, any_pb2.Any())
#
#        operation = tf2_dataset_operation.DatasetWrapperOperation(params_proto)
#        output_dataset = operation.process(self._input_dataset)
#
#        output_list = []
#        for data in output_dataset:
#            output_list.append(data.numpy())
#        actual_output = tf.constant(output_list)
#
#        expected_output = tf.constant([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
#        self.assertAllClose(expected_output, actual_output)
#
#
#class DictToTupleDatasetOperationTest(tf.test.TestCase):
#    """A class for unit-testing the DictToTuple class."""
#    @classmethod
#    def setUpClass(cls):
#        cls._integer_data = tf.constant([0, 1, 2, 3, 4, 5],
#                                        dtype=tf.dtypes.int16)
#        cls._float_data = tf.constant([0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
#                                      dtype=tf.dtypes.float32)
#
#    def test_convert_from_float32_to_int8(self):
#        params_proto = text_format.Parse(
#            """
#            [type.googleapi.com/learning.DictToTupleDatasetOperationParams] {
#                as_supervised: False
#                in_type: FLOAT32
#                out_type: INT8
#                float_max_to_one: True
#            }
#        """, any_pb2.Any())
#
#        dataset = tf.data.Dataset.from_tensor_slices(self._float_data)
#        op = tf2_dataset_operation.DictToTupleDatasetOperation(params_proto)
#        dataset = op.process(dataset)
#
#        output_list = []
#        for data in dataset:
#            output_list.append(data.numpy())
#        actual_output = tf.constant(output_list)
#
#        expected_output = tf.cast(self._float_data * (2**7),
#                                  dtype=tf.dtypes.int8)
#        self.assertAllEqual(expected_output, actual_output)
#
#
#class DictToTupleDatasetOperationTest(tf.test.TestCase):
#    """A class for unit-tesing the DictToTupleDatasetOperation class."""
#    @classmethod
#    def setUpClass(cls):
#        """Prepares input data to be used in the unit test."""
#        # yapf: disable
#        cls._dict_data = {}
#        cls._dict_data["a"] = tf.constant([ 0,  1,  2,  3,  4])
#        cls._dict_data["b"] = tf.constant([10, 11, 12, 13, 14])
#        cls._dict_data["c"] = tf.constant([20, 21, 22, 23, 24])
#        cls._dict_data["d"] = tf.constant([30, 31, 32, 33, 34])
#        cls._dict_data["e"] = tf.constant([-1, -2, -3, -4, -5])
#        # yapf: enable
#
#        cls._input_dataset = tf.data.Dataset.from_tensor_slices(cls._dict_data)
#
#    def test_all_values_case(self):
#        # yapf: disable
#        params_proto = text_format.Parse("""
#            [type.googleapi.com/learning.DictToTupleDatasetOperationParams] {
#                all_values: {
#                    keys: "b"
#                    keys: "d"
#                }
#            }
#        """, any_pb2.Any())
#        # yapf: enable
#
#        op = tf2_dataset_operation.DictToTupleDatasetOperation(params_proto)
#        actual_dataset = op.process(self._input_dataset)
#
#        # Creates the expected data set by selecting the key "b" and "d"
#        dict_expected = tuple(self._dict_data[key] for key in ["b", "d"])
#        expected_dataset = tf.data.Dataset.from_tensor_slices(dict_expected)
#
#        for (expected, actual) in zip(expected_dataset, actual_dataset):
#            self.assertAllEqual(expected, actual)
#
#    def test_dict_inputs_case(self):
#        # yapf: disable
#        params_proto = text_format.Parse("""
#            [type.googleapi.com/learning.DictToTupleDatasetOperationParams] {
#                dict_inputs: {
#                    inputs_keys: "b"
#                    inputs_keys: "d"
#                    targets_key: "e"
#                }
#            }
#        """, any_pb2.Any())
#        # yapf: enable
#
#        op = tf2_dataset_operation.DictToTupleDatasetOperation(params_proto)
#        actual_dataset = op.process(self._input_dataset)
#
#        # Creates the expected data set by selecting the key "b" and "d"
#        dict_expected = tuple(self._dict_data[key] for key in ["b", "d"])
#        expected_dataset = tf.data.Dataset.from_tensor_slices(({
#            "b":
#            self._dict_data["b"],
#            "d":
#            self._dict_data["d"]
#        }, self._dict_data["e"]))
#
#        for (expected, actual) in zip(expected_dataset, actual_dataset):
#            self.assertAllEqual(expected, actual)
#
#
#class UtteranceDataPreoprocessorTest(tf.test.TestCase):
#    """A class for unit-tesing the UtteranceDataPreprocessor class"""
#
#    # TODO TODO(chanw.com) Adds comments and docstrings
#    class DummyFeatureExt(operation.AbstractOperation):
#        """A class mimicking a simple feature extraction.
#
#        In this class, instead of doing feature extraction, 2.0 is simply added
#        to the input sequences.
#        """
#        def __init__(self,
#                     params_proto,
#                     params_dict=None,
#                     transform_dict=None) -> None:
#            pass
#
#        def process(self, inputs: dict):
#            """Adds 2.0 to the input bach of sequence data."""
#            outputs = {}
#            mask = tf.sequence_mask(inputs["SEQ_LEN"],
#                                    maxlen=tf.shape(inputs["SEQ_DATA"])[-1],
#                                    dtype=tf.dtypes.float32)
#
#            outputs["SEQ_DATA"] = tf.math.add(inputs["SEQ_DATA"], 2.0) * mask
#
#            for key in inputs.keys():
#                if key != "SEQ_DATA":
#                    outputs[key] = inputs[key]
#
#            return outputs
#
#    class DummyLabelProcessing(operation.AbstractOperation):
#        def __init__(self,
#                     params_proto,
#                     params_dict=None,
#                     transform_dict=None):
#            pass
#
#        def process(self, inputs):
#            """Splits the input text sequences."""
#            outputs = {}
#
#            outputs["SEQ_DATA"] = tf.strings.split(inputs["SEQ_DATA"])
#            outputs["SEQ_LEN"] = tf.shape(outputs["SEQ_DATA"])[0]
#
#            return outputs
#
#    def test_the_basic_case(self):
#        tf.data.experimental.enable_debug_mode()
#
#        # yapf: disable
#        inputs = (
#            {
#                "SEQ_DATA": tf.constant([
#                    [ 0.0,  1.0,  0.0,  0.0],
#                    [10.0, 11.0, 12.0,  0.0],
#                    [-1.0, -2.0, -3.0, -4.0]
#                ], dtype=tf.dtypes.float32),
#                "SEQ_LEN": tf.constant(
#                    [2, 3, 4], dtype=tf.dtypes.int32),
#                "SEQ_SAMPLING_RATE_HZ": tf.constant(
#                    [1.0, 2.0, 3.0], dtype=tf.dtypes.float32)
#            },
#            {
#                "SEQ_DATA": tf.constant(
#                    ["HELLO BIXBY", "VIDEO MUSIC", "GOOD MUSIC"],
#                    dtype=tf.dtypes.string),
#                 "SEQ_LEN": tf.constant(
#                    [1, 1, 1], dtype=tf.dtypes.int32)
#            },
#        )
#
#        params_proto = text_format.Parse("""
#            [type.googleapi.com/learning.UtteranceDataPreprocessorParams] {
#                audio_processing_operation: {
#                    class_name: "DummyFeatureExt"
#                }
#
#                label_processing_operation: {
#                    class_name: "DummyLabelProcessing"
#                }
#            }
#        """, any_pb2.Any())
#
#        expected_dataset = tf.data.Dataset.from_tensor_slices(
#            ({
#                "SEQ_DATA": tf.constant([
#                    [ 2.0,  3.0,  0.0,  0.0],
#                    [12.0, 13.0, 14.0,  0.0],
#                    [ 1.0,  0.0, -1.0, -2.0]
#                ], dtype=tf.dtypes.float32),
#                "SEQ_LEN": tf.constant(
#                    [2, 3, 4], dtype=tf.dtypes.int32),
#             },
#             {
#                "SEQ_DATA": tf.constant([
#                    ["HELLO", "BIXBY"],
#                    ["VIDEO", "MUSIC"],
#                    ["GOOD",  "MUSIC"]
#                ], dtype=tf.dtypes.string),
#                "SEQ_LEN": tf.constant([2, 2, 2], dtype=tf.dtypes.int32)
#             }))
#        # yapf: enable
#
#        op = tf2_dataset_operation.UtteranceDataPreprocessor(params_proto)
#
#        input_dataset = tf.data.Dataset.from_tensor_slices(inputs)
#
#        actual_dataset = op.process(input_dataset)
#
#        for (expected, actual) in zip(expected_dataset, actual_dataset):
#            # yapf: disable
#            self.assertAllEqual(expected[0]["SEQ_DATA"], actual[0]["SEQ_DATA"])
#            self.assertAllEqual(expected[0]["SEQ_LEN"],  actual[0]["SEQ_LEN"])
#            self.assertAllEqual(expected[1]["SEQ_DATA"], actual[1]["SEQ_DATA"])
#            self.assertAllEqual(expected[1]["SEQ_LEN"],  actual[1]["SEQ_LEN"])
#            # yapf: enable


if __name__ == "__main__":
    tf.test.main()
