import tensorflow as tf

from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes

from tensorflow.python.eager import context
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import InputSpec
from tensorflow.python.ops import nn_ops, standard_ops

class TiedDense(tf.keras.layers.Dense):
    def __init__(self, units, tied_to=None, transpose_kernel=True, **kwargs):
        super(TiedDense, self).__init__(units, **kwargs)
        self.tied_to = tied_to
        self.transpose_kernel = transpose_kernel

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating '
                            'point dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        if isinstance(self.tied_to, tf.keras.layers.Dense):
            self.tied_kernel = self.tied_to.kernel
        elif isinstance(self.tied_to, tf.keras.layers.Embedding):
            self.tied_kernel = self.tied_to.embeddings
        else:
            raise NotImplementedError('invalid tying layer')

        if self.transpose_kernel and (self.tied_kernel.shape[0] != self.units or
                                    self.tied_kernel.shape[1] != input_shape[-1]):
            raise ValueError('Shape of tied weights doesn\'t match ')
        if not self.transpose_kernel and (self.tied_kernel.shape[1] != self.units or
                                    self.tied_kernel.shape[0] != input_shape[-1]):
            raise ValueError('Shape of tied weights doesn\'t match ')

        if self.use_bias:
            self.bias = self.add_weight('bias',
                                        shape=[self.units,],
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        if self.transpose_kernel:
            kernel = tf.transpose(self.tied_kernel)
        else:
            kernel = self.tied_kernel

        # for mixed precision training
        if kernel.dtype !=  self._compute_dtype_object:
            kernel = tf.cast(kernel, self._compute_dtype_object)

        if inputs.dtype !=  self._compute_dtype_object:
            inputs = tf.cast(inputs, self._compute_dtype_object)

        # The code below is from tensorflow-2.5.0.keras.layers.Dense
        # https://github.com/tensorflow/tensorflow/blob/r2.5/tensorflow/python/keras/layers/core.py#L1245-L1257

        outputs = standard_ops.tensordot(inputs, kernel, [[-1], [0]])
        # Reshape the output back to the original ndim of the input.
        if not context.executing_eagerly():
            shape = inputs.shape.as_list()
            output_shape = shape[:-1] + [kernel.shape[-1]]
            outputs.set_shape(output_shape)

        if self.use_bias:
            outputs = nn_ops.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs
