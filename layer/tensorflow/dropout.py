"""A module implementing the RandomProbDroupout layer.

The following classes are implemented.

  * Bypass
  * BaselineDropout
  * UniformDistDropout
  * TwoPointDistDropout
"""
# TODO(chanw.com) Merge masking_layer.py with this module.

# pylint: disable=no-member, import-error

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

# Standard imports
import abc
from enum import Enum

# Third-party imports
import tensorflow as tf
import tensorflow_probability as tfp
from packaging import version

# Custom imports
from math_lib.operation import util
from speech.trainer.ck_trainer.util import proto_util
from machine_learning.layers import dropout_params_pb2

assert version.parse(tf.__version__) >= version.parse("2.0.0"), (
    "At least tensorflow 2.0 is required.")


# TODO(chanw.com) Combine this with _get_dim in
# //machine_learning/losses/seq_util.py.
def _get_dim(tensor, i):
    """Get value of tensor shape[i] preferring static value if available."""
    return tf.compat.dimension_value(tensor.shape[i]) or tf.shape(tensor)[i]


class DropoutFactory(object):
    """A factory class to create a concrete Subsampling layer."""
    def __init__(self) -> None:
        # Creates a dict containing all the classes derived from Subsampling.
        #
        # Note that Subsampling is an "Abstract" class defined in the same
        # module below.
        self._layer_dict = util.create_sub_cls_dict(AbstractDropout)

    def create(self, params_proto: dropout_params_pb2.DropoutParams
            ) -> tf.keras.layers.Layer: # yapf: disable
        assert isinstance(params_proto, dropout_params_pb2.DropoutParams), (
            "The type of \"params_proto\" should be SubsamplingParams.")

        DEFAULT_CLASS_NAME = "BaselineDropout"
        class_name = proto_util.get_field(params_proto, "class_name",
                                          DEFAULT_CLASS_NAME)

        return self._layer_dict[class_name](params_proto)


class AbstractDropout(abc.ABC):
    """An abstract class for applying different types of Dropout."""
    @abc.abstractmethod
    def __init__(self,
                 params_proto: dropout_params_pb2.DropoutParams,
                 **kwargs) -> None: # yapf: disable
        """Initializes an AbstractDropout object.

        Returns:
            None.
        """
        super(AbstractDropout, self).__init__()
        assert isinstance(params_proto, dropout_params_pb2.DropoutParams), (
            "The type of \"params_proto\" should be DropoutParams.")

        DEFAULT_DROPOUT_BYPASS_NUM_EXAMPLES = 0

        self._dropout_bypass_num_examples = tf.convert_to_tensor(
            proto_util.get_field(params_proto, "dropout_bypass_num_examples",
                                 DEFAULT_DROPOUT_BYPASS_NUM_EXAMPLES),
            dtype=tf.dtypes.float64)

        self._seq_noise_shape_type = proto_util.get_field(
            params_proto, "seq_noise_shape",
            dropout_params_pb2.DropoutParams.NONE)

    @property
    def seq_noise_shape_type(self):
        return self._seq_noise_shape_type

    @property
    def dropout_bypass_num_examples(self):
        return self._dropout_bypass_num_examples


class BaselineDropout(tf.keras.layers.Dropout, AbstractDropout):
    """A class performing the baseline dropout.

    Note that this class is directly derived from tf.keras.layers.Dropout.
    """
    def __init__(self,
                 params_proto: dropout_params_pb2.DropoutParams,
                 **kwargs) -> None: # yapf: disable
        assert isinstance(params_proto, dropout_params_pb2.DropoutParams), (
            "The type of \"params_proto\" should be DropoutParams.")
        assert params_proto.class_name == self.__class__.__name__, (
            f"\"class_name\" must be {self.__class__.__name__}.")

        try:
            self._class_params = proto_util.maybe_unpack(
                params_proto.class_params,
                dropout_params_pb2.BaselineDropoutParams)
        except:
            raise ValueError(
                "The \"class_params\" field must be an any params proto "
                "packing an object of the type BaselineDropoutParams.")

        AbstractDropout.__init__(self, params_proto)

        if (self._seq_noise_shape_type ==
                dropout_params_pb2.DropoutParams.BATCH_TIME):
            noise_shape = (1, 1, None)
        elif (self._seq_noise_shape_type ==
              dropout_params_pb2.DropoutParams.BATCH):
            noise_shape = (1, None, None)
        elif (self._seq_noise_shape_type ==
              dropout_params_pb2.DropoutParams.TIME):
            noise_shape = (None, 1, None)
        elif (self._seq_noise_shape_type ==
              dropout_params_pb2.DropoutParams.HIDDEN_FEATURE):
            noise_shape = (None, None, 1)
        else:
            noise_shape = None

        super(BaselineDropout,
              self).__init__(rate=self._class_params.dropout_rate,
                             noise_shape=noise_shape,
                             **kwargs)

    def call(self, inputs, training=None, num_examples=0):
        # If training is False, this layer is bypassed.
        if not training:
            return inputs

        outputs = tf.cond(
            tf.math.greater_equal(
                tf.cast(num_examples, dtype=tf.dtypes.float64),
                self._dropout_bypass_num_examples),
            lambda: super(BaselineDropout, self).call(inputs, training),
            lambda: tf.nest.map_structure(tf.identity, inputs))

        return outputs


# TODO(chanw.com) Use noise shape instead of maintain_pattern
# Refer to the Keras Dropout API for more detail about noise shape.
# This method is similar to _dropout in "masking_layer.py" However, one
# difference is that here, drop_prob is a vector whose size is the same as the
# batch size. Consider refactor these two methods in one command method.
def _dropout(inputs, drop_prob, maintain_pattern=False):
    """Performs the random dropout.

    Args:
        inputs: An input tensor of rank 3.
            The shape is (batch_size, seq_len, feat_dim).
        drop_prob: An input tensor of rank 1.
            The shape should be the same as batch_size of "inputs".

    Returns:
        The masked input.
    """
    if maintain_pattern:
        # In case of an LSTM, the rank must be 3.
        sample_shape = (1, tf.shape(inputs)[-1])
    else:
        sample_shape = tf.shape(inputs)[1:]

    # yapf: disable
    masking = tf.cast(
        tfp.distributions.Bernoulli(
            probs=1.0 - drop_prob).sample(sample_shape),
        dtype=inputs.dtype)
    # yapf: enable
    # The result of tf.roll(tf.range(tf.rank(inputs)), shift=1, axis=0) is as
    # follows:
    #
    # If the rank is 3, then it becomes [2, 0, 1].
    # If the rank is 4, then the result becomes [3, 0, 1, 2].
    # yapf: disable
    masking = tf.transpose(
        masking, perm=tf.roll(tf.range(tf.rank(inputs)), shift=1, axis=0))
    # yapf: enable

    return inputs * masking


def uniform_dist_dropout(inputs, rates, maintain_pattern=False):
    """Performs dropout with a rate from the uniform distribution.

    Args:
        inputs: An input tensor of rank 3.
            The shape is (batch_size, seq_len, feat_dim).
        rates: A tensor containing the min and max of uniform distribution.
            It is a rank one tensor with the shape of [2,] The min and max
            values must be between 0.0 and 1.0. (0.0 <= min <= max <= 1.0).
        maintain_pattern: If True, masking remains the same along the time axis.
            This option is valid only when the input dimension is three
            with the following shape:
                (batch_size, time_steps, feature_size).

    Returns:
        The masked input.
    """
    # TODO(chanw.com) maintain pattern to noise shape.
    if maintain_pattern:
        noise_shape = (1, None)
    else:
        noise_shape = None

    batch_size = _get_dim(inputs, 0)

    # Creates a random vector whose size is equal to "batch_size".
    rates = tf.random.uniform([batch_size],
                              rates[0],
                              rates[1],
                              dtype=inputs.dtype)

    return tf.vectorized_map(lambda x: tf.nn.dropout(*x, noise_shape),
                             [inputs, rates])


class BatchProbDropout(tf.keras.layers.Layer, AbstractDropout):
    def __init__(self,
                 params_proto: dropout_params_pb2.DropoutParams,
                 **kwargs) -> None: # yapf: disable
        assert isinstance(params_proto, dropout_params_pb2.DropoutParams), (
            "The type of \"params_proto\" should be DropoutParams.")
        assert params_proto.class_name == self.__class__.__name__, (
            f"\"class_name\" must be {self.__class__.__name__}.")

        try:
            self._class_params = proto_util.maybe_unpack(
                params_proto.class_params,
                dropout_params_pb2.BatchProbDropoutParams)
        except:
            raise ValueError(
                "The \"class_params\" field must be an any params proto "
                "packing an object of the type BatchProbDropoutParams.")

        AbstractDropout.__init__(self, params_proto)

        super(BatchProbDropout, self).__init__(**kwargs)

        if ((self._seq_noise_shape_type
             == dropout_params_pb2.DropoutParams.BATCH_TIME)
                or (self._seq_noise_shape_type
                    == dropout_params_pb2.DropoutParams.BATCH)):
            # Note that in this dropout approach, we are giving dropout
            # probabilities for each batch independently. However, BATCH_TIME
            # or BATCH means that the pattern is the same across different
            # examples in a single batch. Thus, these noise types do not make
            # sense.
            raise ValueError(
                "The BATCH or BATCH_TIME type does not make sense for "
                "BatchProbDropout.")

        # TODO(chanw.com) Consider supporting this feature in the future.
        if (self._seq_noise_shape_type ==
                dropout_params_pb2.DropoutParams.HIDDEN_FEATURE):
            raise ValueError("The HIDDEN_FEATURE is not supported yet.")
        if (self._seq_noise_shape_type == dropout_params_pb2.DropoutParams.TIME
            ):
            self._maintain_pattern = True
        else:
            self._maintain_pattern = False

    def call(self, inputs, drop_prob, training=None):
        """Runs the UniformDistDropout layer.

        Args:
            inputs: The input Tensor.
                The first axis corresponds to the batch.
            training: A flag representing called during Training.

        Returns:
            An output tensor from this layer.
        """
        if self._maintain_pattern:
            tf.debugging.assert_equal(tf.rank(inputs), 3)

        # If training is False, this layer is bypassed.
        if not training:
            return inputs

        batch_size = tf.shape(inputs)[0]
        outputs = _dropout(inputs, drop_prob, self._maintain_pattern)

        # new_shape is [batch_size, 1, 1, 1]. The rank is the same as the
        # original inputs.
        new_shape = tf.concat(
            [[batch_size],
             tf.ones(tf.rank(inputs) - 1, dtype=tf.dtypes.int32)],
            axis=0)

        # Applies the scaling operation used in the inverted dropout approach.
        return tf.reshape(tf.math.divide_no_nan(1.0, 1.0 - drop_prob),
                          new_shape) * outputs


class UniformDistDropout(tf.keras.layers.Layer, AbstractDropout):
    """A keras layer implementation of UniformDistDropout layer.

    The dropout out probability is selected by a uniform distribution between
    drop_prob_min and drop_prob_max. If both drop_prob_min and drop_prob_max
    are the same, this layer is exactly the same as the normal dropout.
    """
    def __init__(self,
                 params_proto: dropout_params_pb2.DropoutParams,
                 **kwargs) -> None: # yapf: disable
        """Initializes a UniformDistDropout layer.

        Args:
            drop_prob_min: The minimum dropout probability.
            drop_prob_max: The maximum dropout probability.
            maintain_pattern: If True, the mask pattern is unchanged along time.
                This option is valid only when the input dimension is three
                with the following shape:
                    (batch_size, time_steps, feature_size).
        Returns:
            None.
        """
        assert isinstance(params_proto, dropout_params_pb2.DropoutParams), (
            "The type of \"params_proto\" should be DropoutParams.")
        assert params_proto.class_name == self.__class__.__name__, (
            f"\"class_name\" must be {self.__class__.__name__}.")

        try:
            self._class_params = proto_util.maybe_unpack(
                params_proto.class_params,
                dropout_params_pb2.UniformDistDropoutParams)
        except:
            raise ValueError(
                "The \"class_params\" field must be an any params proto "
                "packing an object of the type UniformDistDropoutParams.")

        AbstractDropout.__init__(self, params_proto)
        super(UniformDistDropout, self).__init__(**kwargs)

        if self._class_params.HasField("dropout_rate"):
            dropout_rate = self._class_params.dropout_rate
            assert dropout_rate <= 0.5, (
                "dropout_rate should be equal to or less than 0.5.")
            self._drop_prob_min = 0.0
            self._drop_prob_max = 2.0 * dropout_rate
        elif self._class_params.HasField("bounds"):
            self._drop_prob_min = self._class_params.bounds.min_bound
            self._drop_prob_max = self._class_params.bounds.max_bound
        else:
            self._drop_prob_min = 0.0
            self._drop_prob_max = 0.0

        assert self._drop_prob_min <= self._drop_prob_max, (
            "drop_prob_min should be equal to or less than drop_prob_max.")
        assert self._drop_prob_min >= 0.0 and self._drop_prob_max <= 1.0, (
            "The prob_min and prob_max must be in the interval [0.0, 1.0].")

        if (self._seq_noise_shape_type == dropout_params_pb2.DropoutParams.TIME
            ):
            self._maintain_pattern = True
        elif (self._seq_noise_shape_type ==
              dropout_params_pb2.DropoutParams.NONE):
            self._maintain_pattern = False
        else:
            raise ValueError("Unsupported SeqNoiseShapeType.")

    def call(self, inputs, training=None, num_examples=0):
        """Runs the UniformDistDropout layer.

        Args:
            inputs: The input Tensor.
                The first axis corresponds to the batch.
            training: A flag representing called during Training.

        Returns:
            An output tensor from this layer.
        """
        # If training is False, this layer is bypassed.
        if not training:
            return inputs

        if self._maintain_pattern:
            tf.debugging.assert_equal(tf.rank(inputs), 3)

        outputs = tf.cond(
            tf.math.greater_equal(
                tf.cast(num_examples, dtype=tf.dtypes.float64),
                self._dropout_bypass_num_examples),
            lambda: uniform_dist_dropout(inputs, [
                self._drop_prob_min, self._drop_prob_max
            ], self._maintain_pattern),
            lambda: tf.nest.map_structure(tf.identity, inputs))

        return outputs


class TwoPointDistDropout(tf.keras.layers.Layer, AbstractDropout):
    """A keras layer implementation of TwoPointDistDropout layer.

    The dropout out probability is selected by a uniform distribution between
    drop_prob_min and drop_prob_max. If both drop_prob_min and drop_prob_max
    are the same, this layer is exactly the same as the normal dropout.
    """
    def __init__(self,
                 average_drop_prob=0.0,
                 drop_prob_max=0.5,
                 maintain_pattern=True,
                 **kwargs) -> None:
        """Initializes a TwoPointDistDropout layer.

        Args:
            drop_prob_min: The minimum dropout probability.
            drop_prob_max: The maximum dropout probability.
            maintain_pattern: If True, masking remains the same across time.
                This option is valid only when the input dimension is three
                with the following shape:
                    (batch_size, time_steps, feature_size).
        Returns:
            None.
        """
        super(TwoPointDistDropout, self).__init__(**kwargs)

        assert average_drop_prob <= drop_prob_max, (
            "average_drop_prob should be equal to or less than drop_prob_max.")
        assert average_drop_prob >= 0.0 and drop_prob_max <= 1.0, (
            "The average_drop_prob and drop_prob_max must be in the interval "
            "[0.0, 1.0].")
        assert drop_prob_max > 0, ("drop_prob_max=0.0 should be positive.")

        self._average_drop_prob = average_drop_prob
        self._drop_prob_max = drop_prob_max
        self._prob_max_selection_prob = average_drop_prob / drop_prob_max

        assert self._prob_max_selection_prob <= 1.0, (
            "prob_max_selection_prob cannot be larger than 1.0.")
        self._maintain_pattern = maintain_pattern

        # TODO TODO(chanw.com) Check the minimum version
        self._rand_generator = tf.random.Generator.from_seed(seed=0)

    def call(self, inputs, training=None):
        """Runs the TwoPoint dropout layer.

        Args:
            inputs: The input Tensor.
                The first axis corresponds to the batch.
            training: A flag representing called during Training.

        Returns:
            An output tensor from this layer.
        """
        if self._maintain_pattern:
            tf.debugging.assert_equal(tf.rank(inputs), 3)

        # If training is False, this layer is bypassed.
        if not training or self._average_drop_prob == 0.0:
            return inputs

        batch_size = tf.shape(inputs)[0]
        drop_prob = self._drop_prob_max * self._rand_generator.binomial(
            shape=[batch_size],
            counts=1.0,
            probs=self._prob_max_selection_prob,
            dtype=inputs.dtype)

        outputs = _dropout(inputs, drop_prob, self._maintain_pattern)

        # new_shape is [batch_size, 1, 1, 1]. The rank is the same as the
        # original inputs.
        new_shape = tf.concat(
            [[batch_size],
             tf.ones(tf.rank(inputs) - 1, dtype=tf.dtypes.int32)],
            axis=0)

        # Applies the scaling operation used in the inverted dropout approach.
        return tf.reshape(tf.math.divide_no_nan(1.0, 1.0 - drop_prob),
                          new_shape) * outputs
