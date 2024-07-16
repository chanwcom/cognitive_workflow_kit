"""A module for unit-testing classes in the dropout module."""

# pylint: disable=no-member, import-error

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

# Standard imports
import copy
import os

# Third-party imports
import tensorflow as tf
from google.protobuf import text_format
from packaging import version

# Custom imports
from machine_learning.layers import dropout
from machine_learning.layers import dropout_params_pb2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

assert version.parse(tf.__version__) >= version.parse("2.0.0"), (
    "At least tensorflow 2.0 is required.")


class BatchProbDropoutFactoryTest(tf.test.TestCase):

    def setUp(self):
        # yapf: disable
        self._inputs = tf.constant(
            [[[ 0.0,  1.0,  2.0,  3.0,  4.0],
              [ 5.0,  6.0,  7.0,  8.0,  9.0],
              [10.0, 11.0, 12.0, 13.0, 14.0],
              [15.0, 16.0, 17.0, 18.0, 19.0]],
             [[ 0.5,  1.5,  2.5,  3.5,  4.5],
              [ 5.5,  6.5,  7.5,  8.5,  9.5],
              [10.5, 11.5, 12.5, 13.5, 14.5],
              [15.5, 16.5, 17.5, 18.5, 19.5]]], dtype=tf.dtypes.float32)
        # yapf: enable
        tf.random.set_seed(0)

    def test_baseline_dropout_noise_type_TIME(self):
        # yapf: disable
        SEQ_NOISE_SHAPE_TYPE = dropout_params_pb2.DropoutParams.TIME
        params_proto = text_format.Parse(
            f"""
            seq_noise_shape: {SEQ_NOISE_SHAPE_TYPE}
            class_name:  "BaselineDropout"
            class_params: {{
                [type.googleapi.com/learning.BaselineDropoutParams] {{
                    dropout_rate: 0.5
                }}
            }}
            """, dropout_params_pb2.DropoutParams())
        # yapf: enable
        factory = dropout.DropoutFactory()
        layer = factory.create(params_proto)

        training = True
        actual_output = layer(self._inputs, training)

        # yapf: disable
        expected_output = tf.constant(
            [[ [0.0,  0.0,  4.0,  6.0, 0.0],
              [ 0.0,  0.0, 14.0, 16.0, 0.0],
              [ 0.0,  0.0, 24.0, 26.0, 0.0],
              [ 0.0,  0.0, 34.0, 36.0, 0.0]],
             [[ 1.0,  0.0,  5.0,  7.0, 0.0],
              [11.0,  0.0, 15.0, 17.0, 0.0],
              [21.0,  0.0, 25.0, 27.0, 0.0],
              [31.0,  0.0, 35.0, 37.0, 0.0]]], dtype=tf.dtypes.float32)
        # yapf: enable

        self.assertAllClose(expected_output, actual_output)


class BaselineDropoutTest(tf.test.TestCase):
    """A class for unit-testing classes dervied from the Subsampling class."""

    def setUp(self):
        # yapf: disable
        self._inputs = tf.constant(
            [[[ 0.0,  1.0,  2.0,  3.0,  4.0],
              [ 5.0,  6.0,  7.0,  8.0,  9.0],
              [10.0, 11.0, 12.0, 13.0, 14.0],
              [15.0, 16.0, 17.0, 18.0, 19.0]],
             [[ 0.5,  1.5,  2.5,  3.5,  4.5],
              [ 5.5,  6.5,  7.5,  8.5,  9.5],
              [10.5, 11.5, 12.5, 13.5, 14.5],
              [15.5, 16.5, 17.5, 18.5, 19.5]]], dtype=tf.dtypes.float32)
        # yapf: enable
        tf.random.set_seed(0)

    def test_noise_type_TIME(self):
        # yapf: disable
        SEQ_NOISE_SHAPE_TYPE = dropout_params_pb2.DropoutParams.TIME
        params_proto = text_format.Parse(
            f"""
            seq_noise_shape: {SEQ_NOISE_SHAPE_TYPE}
            class_name:  "BaselineDropout"
            class_params: {{
                [type.googleapi.com/learning.BaselineDropoutParams] {{
                    dropout_rate: 0.5
                }}
            }}
            """, dropout_params_pb2.DropoutParams())
        # yapf: enable

        layer = dropout.BaselineDropout(params_proto)
        training = True
        actual_output = layer(self._inputs, training)

        # yapf: disable
        expected_output = tf.constant(
            [[[ 0.0,  0.0,  4.0,  6.0, 0.0],
              [ 0.0,  0.0, 14.0, 16.0, 0.0],
              [ 0.0,  0.0, 24.0, 26.0, 0.0],
              [ 0.0,  0.0, 34.0, 36.0, 0.0]],
             [[ 1.0,  0.0,  5.0,  7.0, 0.0],
              [11.0,  0.0, 15.0, 17.0, 0.0],
              [21.0,  0.0, 25.0, 27.0, 0.0],
              [31.0,  0.0, 35.0, 37.0, 0.0]]], dtype=tf.dtypes.float32)
        # yapf: enable

        self.assertAllClose(expected_output, actual_output)

    def test_noise_type_BATCH_TIME(self):
        # yapf: disable
        SEQ_NOISE_SHAPE_TYPE = dropout_params_pb2.DropoutParams.BATCH_TIME
        params_proto = text_format.Parse(
            f"""
            seq_noise_shape: {SEQ_NOISE_SHAPE_TYPE}
            class_name:  "BaselineDropout"
            class_params: {{
                [type.googleapi.com/learning.BaselineDropoutParams] {{
                    dropout_rate: 0.5
                }}
            }}
            """, dropout_params_pb2.DropoutParams())
        # yapf: enable

        layer = dropout.BaselineDropout(params_proto)
        training = True
        actual_output = layer(self._inputs, training)

        # yapf: disable
        expected_output = tf.constant(
            [[[ 0.0,  0.0,  4.0,  6.0, 0.0],
              [ 0.0,  0.0, 14.0, 16.0, 0.0],
              [ 0.0,  0.0, 24.0, 26.0, 0.0],
              [ 0.0,  0.0, 34.0, 36.0, 0.0]],
             [[ 0.0,  0.0,  5.0,  7.0, 0.0],
              [ 0.0,  0.0, 15.0, 17.0, 0.0],
              [ 0.0,  0.0, 25.0, 27.0, 0.0],
              [ 0.0,  0.0, 35.0, 37.0, 0.0]]], dtype=tf.dtypes.float32)
        # yapf: enable

        self.assertAllClose(expected_output, actual_output)

    def test_noise_type_HIDDEN_FEATURE(self):
        # yapf: disable
        SEQ_NOISE_SHAPE_TYPE = dropout_params_pb2.DropoutParams.HIDDEN_FEATURE
        params_proto = text_format.Parse(
            f"""
            seq_noise_shape: {SEQ_NOISE_SHAPE_TYPE}
            class_name:  "BaselineDropout"
            class_params: {{
                [type.googleapi.com/learning.BaselineDropoutParams] {{
                    dropout_rate: 0.5
                }}
            }}
            """, dropout_params_pb2.DropoutParams())
        # yapf: enable

        layer = dropout.BaselineDropout(params_proto)
        training = True
        actual_output = layer(self._inputs, training)

        # yapf: disable
        expected_output = tf.constant(
            [[ [0.0,  0.0,  0.0,  0.0,  0.0],
              [ 0.0,  0.0,  0.0,  0.0,  0.0],
              [20.0, 22.0, 24.0, 26.0, 28.0],
              [30.0, 32.0, 34.0, 36.0, 38.0]],
             [[ 0.0,  0.0,  0.0,  0.0,  0.0],
              [11.0, 13.0, 15.0, 17.0, 19.0],
              [ 0.0,  0.0,  0.0,  0.0,  0.0],
              [31.0, 33.0, 35.0, 37.0, 39.0]]], dtype=tf.dtypes.float32)
        # yapf: enable

        self.assertAllClose(expected_output, actual_output)

    def test_noise_type_zero_dropout_rate(self):
        # yapf: disable
        SEQ_NOISE_SHAPE_TYPE = dropout_params_pb2.DropoutParams.HIDDEN_FEATURE
        params_proto = text_format.Parse(
            f"""
            seq_noise_shape: {SEQ_NOISE_SHAPE_TYPE}
            class_name:  "BaselineDropout"
            class_params: {{
                [type.googleapi.com/learning.BaselineDropoutParams] {{
                    dropout_rate: 0.0
                }}
            }}
            """, dropout_params_pb2.DropoutParams())
        # yapf: enable

        layer = dropout.BaselineDropout(params_proto)
        training = True
        actual_output = layer(self._inputs, training)
        expected_output = self._inputs
        self.assertAllClose(expected_output, actual_output)


class BatchProbDropoutTest(tf.test.TestCase):
    """A class for unit-testing classes dervied from the Subsampling class."""

    def setUp(self):
        tf.random.set_seed(0)
        # yapf: disable
        self._inputs = tf.constant(
            [[[ 0.0,  1.0,  2.0,  3.0,  4.0],
              [ 5.0,  6.0,  7.0,  8.0,  9.0],
              [10.0, 11.0, 12.0, 13.0, 14.0],
              [15.0, 16.0, 17.0, 18.0, 19.0]],
             [[ 0.5,  1.5,  2.5,  3.5,  4.5],
              [ 5.5,  6.5,  7.5,  8.5,  9.5],
              [10.5, 11.5, 12.5, 13.5, 14.5],
              [15.5, 16.5, 17.5, 18.5, 19.5]]], dtype=tf.dtypes.float32)
        # yapf: enable

    def test_basic_case_maintain_pattern(self):
        # yapf: disable
        SEQ_NOISE_SHAPE_TYPE = dropout_params_pb2.DropoutParams.TIME
        params_proto = text_format.Parse(
            f"""
            seq_noise_shape: {SEQ_NOISE_SHAPE_TYPE}
            class_name:  "BatchProbDropout"
            """, dropout_params_pb2.DropoutParams())
        # yapf: enable

        layer = dropout.BatchProbDropout(params_proto)
        training = True
        actual_output = layer(self._inputs, tf.constant([0.4, 0.0]), training)

        # yapf: disable
        expected_output = tf.constant(
            [[ [0.0,  0.0,  3.3333333,  0.0, 0.0],
              [ 0.0,  0.0, 11.6666666,  0.0, 0.0],
              [ 0.0,  0.0, 20.0000000,  0.0, 0.0],
              [ 0.0,  0.0, 28.3333333,  0.0, 0.0]],
             [[ 0.5,  1.5,  2.5,  3.5,  4.5],
              [ 5.5,  6.5,  7.5,  8.5,  9.5],
              [10.5, 11.5, 12.5, 13.5, 14.5],
              [15.5, 16.5, 17.5, 18.5, 19.5]]], dtype=tf.dtypes.float32)
        # yapf: enable

        self.assertAllClose(expected_output, actual_output)

    def test_no_dropout_maintain_pattern(self):
        # yapf: disable
        SEQ_NOISE_SHAPE_TYPE = dropout_params_pb2.DropoutParams.TIME
        params_proto = text_format.Parse(
            f"""
            seq_noise_shape: {SEQ_NOISE_SHAPE_TYPE}
            class_name:  "BatchProbDropout"
            """, dropout_params_pb2.DropoutParams())
        # yapf: enable

        layer = dropout.BatchProbDropout(params_proto)
        training = True
        actual_output = layer(self._inputs, tf.constant([0.0, 0.0]), training)

        expected_output = self._inputs
        self.assertAllClose(expected_output, actual_output)


class UniformDistDropoutTest(tf.test.TestCase):
    """A class for unit-testing classes dervied from the Subsampling class."""

    @classmethod
    def setUpClass(cls):
        SEQ_NOISE_SHAPE_TYPE = dropout_params_pb2.DropoutParams.TIME
        # yapf: disable
        cls._params_proto_bounds = text_format.Parse(
            f"""
            seq_noise_shape: {SEQ_NOISE_SHAPE_TYPE}
            class_name:  "UniformDistDropout"
            class_params: {{
                [type.googleapi.com/learning.UniformDistDropoutParams] {{
                    bounds: {{
                        min_bound: 0.0
                        max_bound: 1.0
                    }}
                }}
            }}
            """, dropout_params_pb2.DropoutParams())
        cls._params_proto_rate = text_format.Parse(
            f"""
            seq_noise_shape: {SEQ_NOISE_SHAPE_TYPE}
            class_name:  "UniformDistDropout"
            class_params: {{
                [type.googleapi.com/learning.UniformDistDropoutParams] {{
                    dropout_rate: 0.0
                }}
            }}
            """, dropout_params_pb2.DropoutParams())
        # yapf: enable

    def setUp(self):
        # yapf: disable
        tf.random.set_seed(0)

    def test_uniform_0p0_1p0_maintain_pattern(self):
        # yapf: disable
        inputs = tf.constant(
            [[[ 0.0,  1.0,  2.0,  3.0,  4.0],
              [ 5.0,  6.0,  7.0,  8.0,  9.0],
              [10.0, 11.0, 12.0, 13.0, 14.0],
              [15.0, 16.0, 17.0, 18.0, 19.0]],
             [[ 0.5,  1.5,  2.5,  3.5,  4.5],
              [ 5.5,  6.5,  7.5,  8.5,  9.5],
              [10.5, 11.5, 12.5, 13.5, 14.5],
              [15.5, 16.5, 17.5, 18.5, 19.5]]], dtype=tf.dtypes.float32)
        # yapf: enable
        layer = dropout.UniformDistDropout(self._params_proto_bounds)
        training = True
        actual_output = layer(inputs, training)

        # yapf: disable
        expected_output = tf.constant(
            [[[0.       , 0.       ,  0.      ,  4.237139,  5.649519],
              [0.       , 0.       ,  0.      , 11.299038, 12.711418],
              [0.       , 0.       ,  0.      , 18.360937, 19.773317],
              [0.       , 0.       ,  0.      , 25.422836, 26.835217]],
             [[0.6301725,  1.890518,  3.150863,  4.411207,  5.671553],
              [6.9318972,  8.192243,  9.452587, 10.712933, 11.973277],
              [13.233623, 14.493967, 15.754313, 17.014658, 18.275002],
              [19.535347, 20.795692, 22.056038, 23.316382, 24.576727]]])
        # yapf: enable

        self.assertAllClose(expected_output, actual_output)

    def test_uniform_0p0_1p0_maintain_pattern_non_bypass(self):
        params_proto_bounds = copy.copy(self._params_proto_bounds)
        params_proto_bounds.dropout_bypass_num_examples = 100

        # yapf: disable
        inputs = tf.constant(
            [[[ 0.0,  1.0,  2.0,  3.0,  4.0],
              [ 5.0,  6.0,  7.0,  8.0,  9.0],
              [10.0, 11.0, 12.0, 13.0, 14.0],
              [15.0, 16.0, 17.0, 18.0, 19.0]],
             [[ 0.5,  1.5,  2.5,  3.5,  4.5],
              [ 5.5,  6.5,  7.5,  8.5,  9.5],
              [10.5, 11.5, 12.5, 13.5, 14.5],
              [15.5, 16.5, 17.5, 18.5, 19.5]]], dtype=tf.dtypes.float32)
        # yapf: enable
        layer = dropout.UniformDistDropout(params_proto_bounds)
        training = True
        actual_output = layer(inputs, training, 200)

        # yapf: disable
        expected_output = tf.constant(
            [[[0.       , 0.       ,  0.      ,  4.237139,  5.649519],
              [0.       , 0.       ,  0.      , 11.299038, 12.711418],
              [0.       , 0.       ,  0.      , 18.360937, 19.773317],
              [0.       , 0.       ,  0.      , 25.422836, 26.835217]],
             [[0.6301725,  1.890518,  3.150863,  4.411207,  5.671553],
              [6.9318972,  8.192243,  9.452587, 10.712933, 11.973277],
              [13.233623, 14.493967, 15.754313, 17.014658, 18.275002],
              [19.535347, 20.795692, 22.056038, 23.316382, 24.576727]]])
        # yapf: enable

        self.assertAllClose(expected_output, actual_output)

    def test_uniform_0p0_1p0_maintain_pattern_bypass(self):
        params_proto_bounds = copy.copy(self._params_proto_bounds)
        params_proto_bounds.dropout_bypass_num_examples = 100

        # yapf: disable
        inputs = tf.constant(
            [[[ 0.0,  1.0,  2.0,  3.0,  4.0],
              [ 5.0,  6.0,  7.0,  8.0,  9.0],
              [10.0, 11.0, 12.0, 13.0, 14.0],
              [15.0, 16.0, 17.0, 18.0, 19.0]],
             [[ 0.5,  1.5,  2.5,  3.5,  4.5],
              [ 5.5,  6.5,  7.5,  8.5,  9.5],
              [10.5, 11.5, 12.5, 13.5, 14.5],
              [15.5, 16.5, 17.5, 18.5, 19.5]]], dtype=tf.dtypes.float32)
        # yapf: enable
        layer = dropout.UniformDistDropout(params_proto_bounds)
        training = True
        actual_output = layer(inputs, training, 50)
        expected_output = inputs

        self.assertAllClose(expected_output, actual_output)

    def test_uniform_0p0_0p0_maintain_pattern(self):
        # yapf: disable
        inputs = tf.constant(
            [[[ 0.0,  1.0,  2.0,  3.0,  4.0],
              [ 5.0,  6.0,  7.0,  8.0,  9.0],
              [10.0, 11.0, 12.0, 13.0, 14.0],
              [15.0, 16.0, 17.0, 18.0, 19.0]],
             [[ 0.5,  1.5,  2.5,  3.5,  4.5],
              [ 5.5,  6.5,  7.5,  8.5,  9.5],
              [10.5, 11.5, 12.5, 13.5, 14.5],
              [15.5, 16.5, 17.5, 18.5, 19.5]]], dtype=tf.dtypes.float32)
        # yapf: enable
        layer = dropout.UniformDistDropout(self._params_proto_rate)
        training = True
        actual_output = layer(inputs, True)

        expected_output = inputs
        self.assertAllClose(expected_output, actual_output)


class TwoPointDistDropout(tf.test.TestCase):

    def setUp(self):
        tf.random.set_seed(0)

    def test_average_drop_prob_0p2(self):
        layer = dropout.TwoPointDistDropout(average_drop_prob=0.4)
        inputs = tf.constant(
            [[[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0],
              [10.0, 11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0, 19.0]],
             [[0.5, 1.5, 2.5, 3.5, 4.5], [5.5, 6.5, 7.5, 8.5, 9.5],
              [10.5, 11.5, 12.5, 13.5, 14.5], [15.5, 16.5, 17.5, 18.5, 19.5]]],
            dtype=tf.dtypes.float32)

        actual_output = layer(inputs, True)

        # yapf: disable
        expected_output = tf.constant(
            [[[0.0,  0.0,  4.0, 0.0,  0.0],
              [0.0,  0.0, 14.0, 0.0,  0.0],
              [0.0,  0.0, 24.0, 0.0,  0.0],
              [0.0,  0.0, 34.0, 0.0,  0.0]],
             [[0.0,  3.0,  0.0, 0.0,  9.0],
              [0.0, 13.0,  0.0, 0.0, 19.0],
              [0.0, 23.0,  0.0, 0.0, 29.0],
              [0.0, 33.0,  0.0, 0.0, 39.0]]], dtype=tf.dtypes.float32)
        # yapf: enable
        self.assertAllClose(expected_output, actual_output)


class GlobalMethodTest(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._inputs = tf.ones(shape=(3, 2, 4), dtype=tf.dtypes.float32)

    def setUp(self):
        tf.random.set_seed(1234)

    def test_uniform_dist_dropout_maintain_pattern_false(self):
        actual_output = dropout.uniform_dist_dropout(self._inputs, [0.0, 1.0],
                                                     False)

        # yapf: disable
        expected_output = tf.constant(
            [[[0.0      , 2.164686 , 0.0     , 2.164686],
              [2.164686 , 0.0      , 0.0     , 2.164686]],
             [[1.573858 , 0.0      , 0.0     , 0.0     ],
              [0.0      , 1.573858 , 1.573858, 1.573858]],
             [[0.0      , 0.0      , 0.0     , 0.0     ],
              [0.0      , 0.0      , 2.390230, 0.0      ]]], dtype=tf.dtypes.float32)
        # yapf: enable

        self.assertAllClose(expected_output, actual_output)

    def test_uniform_dist_dropout_maintain_pattern_true(self):
        actual_output = dropout.uniform_dist_dropout(self._inputs, [0.0, 1.0],
                                                     True)

        # yapf: disable
        expected_output = tf.constant(
            [[[0.0     , 2.164686, 0.0     , 2.164686],
              [0.0     , 2.164686, 0.0     , 2.164686]],
             [[1.573858, 1.573858, 0.0     , 1.573858],
              [1.573858, 1.573858, 0.0     , 1.573858]],
             [[0.0     , 0.0     , 0.0     , 0.0     ],
              [0.0     , 0.0     , 0.0     , 0.0     ]]], dtype=tf.dtypes.float32)
        # yapf: enable

        self.assertAllClose(expected_output, actual_output)


if __name__ == "__main__":
    tf.test.main()
