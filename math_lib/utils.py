#!/usr/bin/python3

# pylint: disable=import-error, no-member

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


# TODO(chanw.com, ky85-kim) Make the unit test.
# TODO(chanw.com, ky85-kim) Update the comments and docstrings.
logger=logging.getLogger(__name__)

def extract_subword_vocab(texts, target_vocab_size):
    """Extracts sub-words from a text corpus

    Args:
        texts (list): List of text corpus filepath
          Every line of each file has the content "<uttid> <text>"
        target_vocab_size (int): Target number of sub-word vocab
          Output vocab size might be different.

    Returns:
        string: Filename of sub-word vocab
    """
    def _generator():
        lines = []
        for text_filepath in texts:
            logger.info("load text corpus from {}".format(text_filepath))
            with open(text_filepath, "r") as f:
                lines += f.readlines()
        for line in lines:
            yield line.split(" ", 1)[1]

    encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        _generator(),
        target_vocab_size=target_vocab_size,
        reserved_tokens=["<s>", "<blank>"])

    out_file = "./vocab_{}".format(encoder.vocab_size)
    encoder.save_to_file(out_file)

    return out_file


def load_dict_file(path):
    with open(path, "r") as f:
        import ast
        contents = f.read()
        dictionary = ast.literal_eval(contents)
    assert isinstance(dictionary, dict)
    return dictionary


def encode_batched_text(text_encoder, refs):
    if len(refs) == 0:
        return tf.constant(
            [[]], dtype=tf.int32), tf.constant([], dtype=tf.int32)

    encoded = tf.ragged.stack(
        [text_encoder.encode(ref.numpy()) for ref in refs])
    if isinstance(encoded, tf.RaggedTensor):
        return encoded.to_tensor(0), encoded.row_lengths()

    return encoded, tf.tile([tf.shape(encoded)[1]], [tf.shape(encoded)[0]])

def set_weights(model, variables):
    """Set weights in model with numpy weights in variable_groups

    Args:
        model (tf.keras.Model
               tf.keras.layers.Layer
               dict<string, model>
               list<model>):
            models we want to assign weights to.
        variables (dict<string, variables>
                   tuple<int, np.array>):
            variable (or variable groups) containing the weights

    * The keys of variables should be keys(or members if model is
    tf.keras.Model) of the model.
    * set_weights() will be called recursively with model.key until we
    can set the model(tf.keras.Layer) with variables(tuple<>).
    * Depending on the type of the layer, different tensors are combined
    to set the weights.
    * The first int valye of variables(tuple<>) is the first dimension of
    the weight saved in np.array. This is set as the last dimension of this
    layer's input tensor if the weight is that of a kernel.

    """
    def _set_layer_weights():
        #print(f'set weights for {model} with keys', list(variables.keys()))
        if 'embeddings' in variables:
            # embedding layer
            shape = tf.TensorShape([None])
            weights = [variables['embeddings']]
        elif 'kernel' in variables:
            # dense layer
            shape = tf.TensorShape([variables['kernel'].shape[0]])
            weights = [variables['kernel'],
                       variables['bias']]
        elif 'cell/kernel' in variables:
            # lstm layer
            shape = tf.TensorShape([None, None, variables['cell/kernel'].shape[0]])
            weights = [variables['cell/kernel'],
                       variables['cell/recurrent_kernel'],
                       variables['cell/bias']]
        else:
            raise ValueError('Unsupported layer')

        model.build(shape)
        model.set_weights(weights)


    if isinstance(model, tf.keras.layers.Layer) and \
       not isinstance(model, tf.keras.Model):
        return _set_layer_weights()

    variable_groups = dict()
    for key, value in variables.items():
        group_key, sub_group_key = key.split('/', 1)
        if group_key in variable_groups:
            variable_groups[group_key][sub_group_key] = value
        else:
            variable_groups[group_key] = {sub_group_key: value}

    for key, sub_groups in variable_groups.items():
        if isinstance(model, dict) and key in model:
            attr = model[key]
        elif isinstance(model, list):
            attr = model[int(key)]
        elif isinstance(model, tf.keras.Model) and hasattr(model, key):
            attr = getattr(model, key)
        else:
            raise ValueError(f'{key} not in {model}')

        set_weights(attr, sub_groups)

def set_average_weights_with_checkpoints(model, checkpoints):
    """Average weights from multiple checkpoints and assign to model

    Args:
        model (dict<string, tf.keras.Model>):
            the model returned from model.get_model_dict()
        checkpoints (list):
            paths of the checkpoints to average

    Returns:
        var_values (dict<string, int, np.array>):
            variable names : (input_shape, variable_tensor)

    Notes:
        This function only assigns variables for evaluation
        TODO: assign optimizer, global_step, save_counter, etc. for training

    """
    def _aggregate_variables(checkpoints):
        var_list = tf.train.list_variables(checkpoints[0])

        var_values = dict()
        for checkpoint in checkpoints:
            logger.info(f'read {checkpoint}')
            reader = tf.train.load_checkpoint(checkpoint)

            for name, shape in var_list:
                if not name.endswith('.ATTRIBUTES/VARIABLE_VALUE') or \
                        '.OPTIMIZER_SLOT' in name or \
                        name.startswith('global_step') or \
                        name.startswith('optimizer') or \
                        name.startswith('save_counter'):
                    continue

                tensor = reader.get_tensor(name)
                name = name.replace('/.ATTRIBUTES/VARIABLE_VALUE', '')
                if name in var_values:
                    var_values[name] += tensor
                else:
                    var_values[name] = tensor

        for name in var_values:
            var_values[name] /= len(checkpoints)

        return var_values

    variables = _aggregate_variables(checkpoints)
    set_weights(model, variables)

def load_checkpoint_as_dict(checkpoint_path):
    variables = {}
    reader = tf.train.load_checkpoint(checkpoint_path)
    for name, shape in tf.train.list_variables(checkpoint_path):
        # exclude optimizer
        if "Adam" in name or ".OPTIMIZER_SLOT" in name:
            continue

        weight = reader.get_tensor(name)

        # simplify name
        name = name.replace("/.ATTRIBUTES/VARIABLE_VALUE", "")

        if isinstance(weight, np.ndarray):
            variables[name] = tf.Variable(weight)

    def make_model_dict(variables):
        model_dict = {}
        for full_name in variables:
            if "/" in full_name:
                key, sub_var = full_name.split("/", maxsplit=1)
                # add to model
                if not key in model_dict:
                    model_dict[key] = {}
                model_dict[key][sub_var] = variables[full_name]
            else:
                model_dict[full_name] = variables[full_name]

        for key in model_dict:
            if isinstance(model_dict[key], dict):
                model_dict[key] = make_model_dict(model_dict[key])

        return model_dict

    return make_model_dict(variables)

def print_checkpoint_dict(d, prefix="", indent="    "):
    for key in d:
        print(f"{prefix}{key}")
        if isinstance(d[key], dict):
            print_checkpoint_dict(d[key], f"{prefix}{indent}")
        if isinstance(d[key], list):
            for i, val in enumerate(d[key]):
                print(f"{prefix}{indent}{i}")
                print_checkpoint_dict(val, f"{prefix}{indent}{indent}")

def get_model_details(model, step=None):
    '''
    Write histogram of all trainable vairables in model.

    Args:
        model: tf.keras.Model or tf_trainer.models.ModelBase
        step:

    '''
    for v in model.trainable_variables:
        tf.summary.histogram(v.name + '_hist', v, step)

def visualize_tensor(x, step=None):
    if len(x.shape) == 2:
        height, width = x.shape
        channel = 1
    else:
        height, width, channel = x.shape

    import numpy as np
    img = np.reshape(x, (-1, height, width, channel))
    tf.summary.image('Image', img, step)

def import_model(model):
    try:
        model_type = model.split(".")[0]
        model_class = model.split(".")[1]

        mod = __import__("speech.trainer.tf_based_end_to_end_trainer."
            "tf_trainer.models." + model_type, fromlist=[model_class])
        Model = getattr(mod, model_class)
        return Model
    except ImportError:
        raise ImportError(f"Unsupported model type {model}")

class NonMirroredStrategy(object):
    """ Non Mirrored Strategy
    This is for replacing open('dev/null') with this class
    """
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_value, tb):
        pass
    def scope():
        return NonMirroredStrategy()

def tf_resize(input_data, shape):
    """
    Args:
      input_data: a `Tensor`
      shape: requested output shape
    """
    input_size = tf.size(input_data)
    output_size = tf.reduce_prod(shape)
    input_data_flatten = tf.reshape(input_data, [input_size])
    result = tf.tile(input_data_flatten, [output_size // input_size + 1])
    return tf.reshape(result[0:output_size], shape=[shape])

def buffered_arange(max_val):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = tf.Variable(0)
    buffered_arange.buf = tf_resize(buffered_arange.buf, max_val)
    return buffered_arange.buf[:max_val]

def compute_mask_indices(
    shape,
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    padding_mask = None,
    require_same_masks = True
) :
    """
    Computes random mask spans for a given shape
    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking
                      padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked.
                   this will be multiplied by number of timesteps divided by length of mask span
                   to mask approximately this percentage of all elements. however due to overlaps,
		   the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other.
		     mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents
		    spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked
		   between spans
        require_same_masks: if true, will randomly drop out masks until same amount of masks
                            remains in each sample
    """
    if type(mask_type) == bytes:
        mask_type = mask_type.decode()
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len and require_same_masks:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return tf.convert_to_tensor(mask)


def _expand_dims_from_front(target, ndim):
    """Insert additional dimension at the front

    Args:
        target(list, tuple or tf.Tensor): Object to be modified
        ndim(int): number of the dimensions to be applied

    Returns:
        The expanded object(with the given target type) with target_ndim rank.
    """
    def _expand_shape(shape, ndim):
        if len(shape) == ndim:
            return shape
        if len(shape) > ndim:
            assert False

        pad = [1, ] * (ndim - len(shape))
        return pad + list(shape)

    def _expand_tensor(tensor, target_ndim):
        t_shape = tensor.shape.as_list()
        to_expand = target_ndim - len(t_shape)
        for _ in range(to_expand):
            tensor = tf.expand_dims(tensor, axis=0)
        return tensor

    if isinstance(target, (list, tuple)):
        return _expand_shape(target, ndim)
    if isinstance(target, tf.Tensor):
        return _expand_tensor(target, ndim)

    assert False


def expand_dims_from_front(target_list, ndim):
    """Insert additional dimension at the front

    Args:
        target_list(list): list of targets(list, tuple or tensor) to be modified
        ndim(int): number of the dimensions to be applied

    Returns:
        The list of expanded shapes.
    """
    if len(target_list) == 1:
        return _expand_dims_from_front(target_list[0], ndim)
    return [_expand_dims_from_front(target, ndim) for target in target_list]


def make_input_signature(spec_tuples, force_4dim=True):
    """Make input_signature for tf.function.

    Args:
        spec_tuples(iterable): Iterable of tuples(shape, dtype) or (shape, dtype, name)
        force_4dim(boolean): Whether to expand dimension at the front
            to make shape 4 dimensional

    Returns:
        A list of tf.TensorSpec with given shapes(reshaped if needed) and dtypes.
    """
    def _parse_spec_tuple(tup):
        if len(tup) == 2:
            shape, dtype = tup
            name = None
        if len(tup) == 3:
            shape, dtype, name = tup
        return shape, dtype, name

    def _make_spec(shape, dtype, name):
        if name is None:
            spec = tf.TensorSpec(shape=shape, dtype=dtype)
        else:
            spec = tf.TensorSpec(shape=shape, dtype=dtype, name=name)
        return spec

    input_signature = []
    for tup in spec_tuples:
        shape, dtype, name = _parse_spec_tuple(tup)
        if force_4dim:
            shape = _expand_dims_from_front(shape, 4)
        input_signature.append(_make_spec(shape, dtype, name))
    return input_signature


def squeeze_dims_from_front(tensors, shapes):
    """Remove dimensions from the first until the tensor has the targeted shape

    Args:
        tensors(tf.Tensor or list of tf.Tensor): tensor(s) to be reshaped
        shapes(list): target shape or list of shapes

    Returns:
        The squeezed tensor or list of tensors
    """
    if isinstance(tensors, (list, tuple)):
        assert len(tensors) == len(shapes)
        return [squeeze_dims_from_front(tensor, shape) for tensor, shape in zip(tensors, shapes)]

    tensor = tensors
    shape = shapes
    assert isinstance(tensor, (tf.Tensor, tf.Variable))
    t_shape = tensor.shape.as_list()
    if len(t_shape) < len(shape):
        assert False
    if len(t_shape) == len(shape):
        return tensor

    to_squeeze = len(t_shape) - len(shape)
    for i in range(to_squeeze):
        assert t_shape[i] == 1
        tensor = tensor[0]

    for i in range(len(shape)):
        assert t_shape[to_squeeze + i] == shape[i]

    return tensor

def get_concrete_function_scope(compat_fp16=False):
    if compat_fp16:
        return 'convert_fp16compat'
    else:
        return 'convert'

def is_fp16_compat_converting():
    cur_name_scope = tf.compat.v1.get_default_graph().get_name_scope()
    if 'convert' in cur_name_scope and 'fp16compat' in cur_name_scope:
        return True
    return False

def early_stopping_call(last_k_loss, inc_tracker, valid_loss_dict, loss_dict, delta, alpha, galpha, criteria, k):
    """
    Reference paper: Automatic early stopping using cross validation: quantifying the criteria
    doi:https://doi.org/10.1016/S0893-6080(98)00010-0

    k --> window size

    last_k_loss[0] --> last k validation losses
    last_k_loss[1] --> last k training losses

    inc_tracker[0] --> no. of consecutive steps for which validation loss has been increasing
    inc_tracker[1] --> minimum validaion loss till time t
    loss_dict --> consists the losses as keys and value as the loss value to be used for early stopping
    criteria --> used to supply appropriate key in loss_dict
    """
    if len(last_k_loss) == 0:
        last_k_loss.append([valid_loss_dict["valid_"+criteria].numpy()])
        last_k_loss.append([loss_dict[criteria].numpy()])
        inc_tracker.append(valid_loss_dict["valid_"+criteria].numpy())
        return False

    if last_k_loss[0][-1] < valid_loss_dict["valid_"+criteria].numpy():
        inc_tracker[0] += 1
    else:
        inc_tracker[0] = 0

    ## if the loss has been increasing for some consecutive steps more than the window size will return true
    if inc_tracker[0] > k:
        logger.info("Stopping early as loss has been increasing for more than {} validation steps".format(k))
        return True

    if len(last_k_loss[0]) < k:
        last_k_loss[0].append(valid_loss_dict["valid_"+criteria].numpy())
        last_k_loss[1].append(loss_dict[criteria].numpy())
        return False

    ## moving the window forward, popping the oldest value and inserting the latest value
    last_k_loss[0][:-1] = last_k_loss[0][1:]
    last_k_loss[0][-1] =  valid_loss_dict["valid_"+criteria].numpy()
    last_k_loss[1][:-1] = last_k_loss[1][1:]
    last_k_loss[1][-1] =  loss_dict[criteria].numpy()

    GL_t = 100*(last_k_loss[0][-1]/inc_tracker[1] - 1)
    Pk_t = 1000*(np.mean(np.array(last_k_loss[1]))/np.min(np.array(last_k_loss[1])) - 1)

    ## In paper GL(t) > 5 gives good tradeoff between processing time and efficiency but this value can excced galpha for a single step
    ## and then start decreasing again. So it is associated with the condition that validation loss should be increasing for at least 2 steps
    if GL_t > galpha and inc_tracker[0] > 1 :
        logger.info("Stopping early as value of GL(t) = {} is greater than galpha provided".format(GL_t))
        return True

    PQ_a = (1.0*GL_t)/Pk_t

    if PQ_a > alpha:
        return True

    inc_tracker[1] = min(inc_tracker[1],last_k_loss[0][-1])
    valid_k_loss = np.array(last_k_loss[0])

    ### if the change in the validation loss has been less than delta than the function should return True
    diff_vals = 1000*np.abs(valid_k_loss/np.mean(valid_k_loss) - 1)
    bool_vals = diff_vals > delta
    if not np.any(bool_vals):
        logger.info("Stopping early as the last k(early_stopping_ws) values have changed (10*percentage) less than delta provided")
        return True
    return False
