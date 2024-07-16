from packaging import version

import tensorflow as tf

from tensorflow.keras.layers import LayerNormalization
from tensorflow.python.training.tracking import data_structures


def get_last_values(x, seq_len, n_output, axis=1):
    """Get x[:, seq_len-n_output:seq_len, :].

    Args:
        x: Tensor, [*batch_shape, T, *shape]
        seq_len: Tensor, [*batch_shape]
        n_output: int
        axis: time-axis

    Return:
        Tensor [*batch_shape, n_output, *shape]
    """
    mask = tf.sequence_mask(seq_len, maxlen=tf.shape(x)[axis], dtype=tf.int32)
    mask_zero = tf.sequence_mask(seq_len - n_output,
                                 maxlen=tf.shape(x)[axis],
                                 dtype=tf.int32)
    last_values = tf.dynamic_partition(x, mask - mask_zero, 2)[1]

    output_shape = shape_list(x)
    output_shape[axis] = n_output
    return tf.reshape(last_values, output_shape)


def shape_list(x, out_type=tf.int32):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x, out_type=out_type)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def flatten_nested_tensors(maybe_nested_tensors):
    """Make flattened list of Tensors

    e.g. [A, [B, [C, D], E], [F, G]] -> [A, B, C, D, E, F, G]

    Args:
        maybe_nested_tensors: Iterable including tf.Tensor or Iterable

    Return:
        List
    """
    unpacked = []
    for element in maybe_nested_tensors:
        if isinstance(element, tf.Tensor):
            unpacked += [
                element,
            ]
        else:
            unpacked += flatten_nested_tensors(element)
    return unpacked


def get_sublayers(layer):
    assert isinstance(layer, tf.keras.layers.Layer)
    if version.parse(tf.__version__) >= version.parse("2.5"):
        return layer._self_tracked_trackables
    else:
        return layer._layers


# prevent the model from using fusedBatchNorm
def disable_fused_ln(layer):
    if isinstance(layer, LayerNormalization):
        layer._fused = False
        return

    if isinstance(layer, (data_structures.ListWrapper, list)):
        for l in layer:
            disable_fused_ln(l)
        return

    if isinstance(layer, tf.keras.layers.Layer):
        disable_fused_ln(get_sublayers(layer))
        return


@tf.custom_gradient
def grad_scale(x, scale):

    def grad(dy):
        return dy * scale, None

    return x, grad
