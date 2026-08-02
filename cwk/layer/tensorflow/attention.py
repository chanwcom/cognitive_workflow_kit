# Standard imports
from itertools import cycle

# Third-party imports
import numpy as np
import tensorflow as tf

# Custom imports
from layer.tensorflow import layer_utils
from layer.tensorflow.tied_layer import TiedDense
from layer.tensorflow.layer_utils import shaped_list
from math_lib import utils


def cast_if_fp16(x):
    if isinstance(x, list):
        result = []
        for _x in x:
            result.append(cast_if_fp16(_x))
        return result

    if x.dtype == tf.float16:
        x = tf.cast(x, tf.float32)
    return x


class RelativePositionEncoding(tf.keras.layers.Layer):
    """ Relative Position Encoding from Transformer-XL

    https://arxiv.org/pdf/1901.02860.pdf
    Modified for variable length sequences.

    Arguments:
        input_dim: input dimension
    """

    def __init__(self, input_dim):
        super().__init__()
        self.inv_freq = tf.Variable(np.array(
            1 / (10000.0**(np.arange(0, input_dim, 2.0) / input_dim))),
                                    dtype=tf.float32,
                                    trainable=False)
        self.input_dim = input_dim

    def call(self, seq_len):
        pos_seq = tf.range(seq_len - 1, -seq_len, -1, dtype=tf.int32)
        pos_seq = tf.cast(pos_seq, tf.float32)
        sinusoid_input = tf.multiply(tf.expand_dims(pos_seq, axis=-1),
                                     tf.expand_dims(self.inv_freq, axis=0))
        sinusoid_input = tf.expand_dims(sinusoid_input, -1)

        pos_embed = tf.concat(
            [tf.math.sin(sinusoid_input),
             tf.math.cos(sinusoid_input)], -1)

        pos_embed_length = self.pos_emb_length(seq_len)
        pos_embed = tf.reshape(pos_embed, [pos_embed_length, self.input_dim])

        return pos_embed

    def pos_emb_length(self, seq_len):
        return 2 * seq_len - 1


def compute_weight_from_energy(energy,
                               x_len,
                               left_mask,
                               right_mask,
                               stream=False,
                               dropout=0.0,
                               valid_state=None,
                               renormalize_dropout=False,
                               dropout_window=1,
                               dropout_column=False):
    """ Apply masking and dropout, take softmax to attention energy (logit).

    energy: [B,N,Q,K] ([B, n_att_head, n_query, n_keys(=n_values)])
    """

    batch_size, num_heads, num_querys, num_keys = shape_list(energy)
    if x_len is None:
        need_tile = False
        seq_mask = tf.ones_like(energy)
    else:
        need_tile = True
        seq_mask = tf.sequence_mask(x_len, maxlen=num_querys)  # [B, K]
        seq_mask = tf.expand_dims(tf.expand_dims(seq_mask, axis=1), axis=1)
        if renormalize_dropout and dropout > 0.0:
            if dropout_column:
                drop_shape = (batch_size, num_heads, 1,
                              num_keys // dropout_window + 1)
            else:
                drop_shape = (batch_size, num_heads, num_querys,
                              num_keys // dropout_window + 1)
            drop_random = tf.random.uniform(drop_shape)
            keep_mask = drop_random >= dropout
            if dropout_column:
                keep_mask = tf.tile(keep_mask, [1, 1, num_querys, 1])
            keep_mask = tf.repeat(keep_mask, repeats=dropout_window, axis=3)

            seq_mask = tf.math.logical_and(seq_mask,
                                           keep_mask[:, :, :, :num_keys])
        seq_mask = tf.cast(seq_mask, tf.float32)

    if stream:
        assert right_mask == 0
        if need_tile:
            seq_mask = tf.broadcast_to(
                seq_mask, [batch_size, num_heads, num_querys, num_keys])

        # Replacement of tf.linalg.band_part(seq_mask, 0, left_mask)
        mask_right = tf.sequence_mask(tf.range(left_mask + 1, num_keys + 1),
                                      dtype=tf.float32)
        mask_left_inverse = tf.sequence_mask(tf.range(num_querys),
                                             maxlen=num_keys,
                                             dtype=tf.float32)
        mask = tf.multiply(seq_mask, mask_right - mask_left_inverse)

        if valid_state is not None:
            mask = tf.concat([
                tf.zeros([tf.math.maximum(0, left_mask - valid_state)]),
                tf.ones([tf.math.minimum(valid_state + 1, left_mask + 1)])
            ], 0) * mask
    elif left_mask == -1 and right_mask == -1:
        if need_tile:
            pass
        mask = seq_mask
    else:
        if need_tile:
            seq_mask = tf.broadcast_to(
                seq_mask, [batch_size, num_heads, num_querys, num_keys])
        mask = tf.linalg.band_part(seq_mask, left_mask, right_mask)

    attention = masked_softmax(energy, mask)
    if not renormalize_dropout and dropout > 0.0:
        attention = tf.nn.dropout(attention, dropout)

    return attention


def unstack_and_matmul(mat1, mat2, transpose_b=False):
    if len(mat1.shape) == 2:
        return tf.matmul(mat1, mat2, transpose_b=transpose_b)

    def _match_dim_zip(l1, l2):
        if len(l1) == 1 and len(l2) != 1:
            return zip(cycle(l1), l2)
        elif len(l1) != 1 and len(l2) == 1:
            return zip(l1, cycle(l2))
        assert len(l1) == len(l2), "Custom Einsum: size not broadcastable"
        return zip(l1, l2)

    result = []
    for sub_mat1, sub_mat2 in _match_dim_zip(tf.unstack(mat1),
                                             tf.unstack(mat2)):
        result.append(
            unstack_and_matmul(sub_mat1, sub_mat2, transpose_b=transpose_b))

    return tf.stack(result)


def einsum(mat1,
           mat2,
           trans1=None,
           trans2=None,
           trans3=None,
           transpose_b=False):
    if trans1 is not None:
        mat1 = tf.transpose(mat1, trans1)
    if trans2 is not None:
        mat2 = tf.transpose(mat2, trans2)

    result = unstack_and_matmul(mat1, mat2, transpose_b=transpose_b)

    if trans3 is not None:
        result = tf.transpose(result, trans3)

    return result


def compute_context(att_weight, value, stream=False, transpose_value=False):
    # att_weight
    #    - [B,N,Q,K] ([B, n_att_head, n_query, n_keys(=n_values)])
    #    - [N,B,K] (when encoder's batch and decoder's timestep are both 1)
    # value [B,K,N,D] or [B,N,D,K] if transpose_value

    if len(att_weight.shape) == 3 and value.shape[0] == 1:
        value = tf.squeeze(value, axis=0)
        context = einsum(att_weight,
                         value,
                         trans2=None if transpose_value else [1, 0, 2],
                         trans3=[1, 0, 2],
                         transpose_b=transpose_value)

        B, N, D = shape_list(context)
        return tf.reshape(context, [B, 1, N * D])
    elif att_weight.shape[0] is not None:
        context = einsum(att_weight,
                         value,
                         trans2=None if transpose_value else [0, 2, 1, 3],
                         trans3=[0, 2, 1, 3],
                         transpose_b=transpose_value)
    else:
        equation = 'bnij,{}->bind'.format(
            'bndj' if transpose_value else 'bjnd')
        context = tf.einsum(equation, att_weight, value)

    # [B,Q,N,D] -> [B,Q,N*D]
    B, Q, N, D = shape_list(context)
    context = tf.reshape(context, [B, Q, N * D])
    return context


def masked_softmax(energy, mask):
    if mask is None:
        return tf.nn.softmax(energy)
    large_number = 6e4 if energy.dtype == tf.float16 else 1e30
    if utils.is_fp16_compat_converting():
        large_number = 6e4
    mask = (1.0 - mask) * large_number
    masked_energy = energy - tf.cast(mask, energy.dtype)

    return tf.nn.softmax(masked_energy)


def sequence_mask(lengths, maxlen, dtype):
    assert lengths.shape[0] is not None
    masks = []
    for length in tf.unstack(lengths):
        masks.append(
            tf.concat([
                tf.ones([length], dtype=dtype),
                tf.zeros([maxlen - length], dtype=dtype)
            ],
                      axis=-1))
    return tf.stack(masks)


class RelativeAttention(tf.keras.layers.Layer):
    """ Relative attention from Transformer-XL

    https://arxiv.org/pdf/1901.02860.pdf
    Modified for variable length sequences.

    Arguments:
        att_dim: the total dimension of final context, note that dimensions of
                 individual attention heads are att_dim // att_head
        att_head: the number of attention heads
        input_dim: the dimension of input
        renormalize_dropout: Enable Normalized Rescaling in
                         DropAttention (https://arxiv.org/pdf/1907.11065.pdf)
        dropout_window: window size in DropAttention
        dropout_column: Use DropAttention(c), randomly drop 'columns(=keys)'

        - dropout_window, dropout_column are used only when renormalize_dropout=True
    """

    def __init__(self,
                 att_dim,
                 att_head,
                 input_dim,
                 left_mask=-1,
                 right_mask=-1,
                 dropout=0.0,
                 renormalize_dropout=False,
                 dropout_window=1,
                 dropout_column=False):
        super().__init__()
        self.num_head = att_head
        self.content_key = tf.keras.layers.Dense(att_dim,
                                                 use_bias=False)  # W_k,E
        self.relative_key = tf.keras.layers.Dense(att_dim,
                                                  use_bias=False)  # W_k,R
        self.content_query = tf.keras.layers.Dense(att_dim,
                                                   use_bias=True)  # W_q, u
        self.relative_query = TiedDense(att_dim,
                                        tied_to=self.content_query,
                                        transpose_kernel=False,
                                        use_bias=True)  # W_q, v
        self.value = tf.keras.layers.Dense(att_dim)  # W_v

        self.head_dim = att_dim // att_head
        self.R = RelativePositionEncoding(input_dim)

        self.left_mask = left_mask
        self.right_mask = right_mask

        self.dropout = dropout
        self.renormalize_dropout = renormalize_dropout
        self.dropout_window = dropout_window
        self.dropout_column = dropout_column

    def call(self, x, x_len, training=False):
        """
        x: [batch, seq, H]
        i: query ([0, seq-1])
        j: key ([0, seq-1])
                   /*   query   */      /*  key  */
        A_i,j = (E_x_i' * W_q' + u') * W_k,E * E_x_j + /* content energy */
                (E_x_i' * W_q' + v') * W_k,R * R_i-j   /* relative energy */
        """
        B, S, _ = shape_list(x)  # S = Q = K

        # query
        q_con = self.content_query(x)  # E_x_i' * W_q' + u'
        q_con = tf.reshape(q_con,
                           [B, S, self.num_head, self.head_dim])  # [B,Q,N,D]
        q_rel = self.relative_query(x)  # E_x_i' * W_q' + v'
        q_rel = tf.reshape(q_rel,
                           [B, S, self.num_head, self.head_dim])  # [B,Q,N,D]

        k_con = self.content_key(x)  # W_k,E * E_x_j
        k_con = tf.reshape(k_con,
                           [B, S, self.num_head, self.head_dim])  # [B,Q,N,D]
        if q_con.shape[0] is not None:
            content_energy = einsum(q_con,
                                    k_con,
                                    trans1=[0, 2, 1, 3],
                                    trans2=[0, 2, 3, 1])
        else:
            content_energy = tf.einsum('bind,bjnd->bnij', q_con,
                                       k_con)  # [B,N,Q,K]

        k_rel = self.relative_key(self.R(S))  # [2*K-1, N*D]
        rel_pos_len = self.R.pos_emb_length(S)
        k_rel = tf.reshape(
            k_rel,
            [rel_pos_len, self.num_head, self.head_dim])  # [2*K-1, N, D]

        if q_rel.shape[0] is not None:
            relative_energy = einsum(q_rel,
                                     tf.expand_dims(k_rel, axis=0),
                                     trans1=[0, 2, 1, 3],
                                     trans2=[0, 2, 3, 1])
        else:
            relative_energy = tf.einsum('bind,jnd->bnij', q_rel,
                                        k_rel)  # [B,N,Q,2*K-1]
        relative_energy = self.rel_shift(
            relative_energy)  # [B,N,Q,2*K-1] -> [B, N, Q, K]

        energy = tf.divide(content_energy + relative_energy,
                           tf.math.sqrt(tf.cast(self.head_dim,
                                                tf.float32)))  # [B,N,Q,K]

        dropout = self.dropout if training else 0
        attention = compute_weight_from_energy(
            energy,
            x_len,
            self.left_mask,
            self.right_mask,
            dropout=dropout,
            renormalize_dropout=self.renormalize_dropout,
            dropout_window=self.dropout_window,
            dropout_column=self.dropout_column)

        v = tf.reshape(self.value(x),
                       [B, S, self.num_head, self.head_dim])  # [B,K,N,D]
        context = compute_context(attention, v)  # [B, Q, N*D]

        return context

    @staticmethod
    def rel_shift(x):
        shape = shape_list(x)  # [B,N,Q,2K-1], Q == K
        x = tf.pad(x, ((0, 0), (0, 0), (0, 0), (1, 0)))  # [B,N,Q,2K]
        x = tf.reshape(x, [shape[0], shape[1], shape[3] + 1, shape[2]])
        x = tf.slice(x, [0, 0, 1, 0], [shape[0], shape[1], shape[3], shape[2]])
        x = tf.reshape(x, shape)
        x = tf.slice(x, [0, 0, 0, 0], [shape[0], shape[1], shape[2], shape[2]])
        return x


class SelfAttention(tf.keras.layers.Layer):
    """ SelfAttention with limited context

    Arguments:
        att_dim: the total dimension of final context, note that dimensions of
                 individual attention heads are att_dim // att_head
        att_head: the number of attention heads
        renormalize_dropout: Enable Normalized Rescaling in
                         DropAttention (https://arxiv.org/pdf/1907.11065.pdf)
        dropout_window: window size in DropAttention
        dropout_column: Use DropAttention(c), randomly drop 'columns(=keys)'

        - dropout_window, dropout_column are used only when renormalize_dropout=True
    """

    def __init__(self,
                 att_dim,
                 att_head,
                 use_bias=True,
                 left_mask=-1,
                 right_mask=-1,
                 use_proj=False,
                 dropout=0.0,
                 renormalize_dropout=False,
                 dropout_window=1,
                 dropout_column=False):
        super().__init__()
        self.num_head = att_head
        self.content_key = tf.keras.layers.Dense(att_dim,
                                                 use_bias=use_bias)  # W_k,E
        self.content_query = tf.keras.layers.Dense(att_dim,
                                                   use_bias=use_bias)  # W_q, u
        self.value = tf.keras.layers.Dense(att_dim, use_bias=use_bias)  # W_v
        self.use_proj = use_proj
        if self.use_proj:
            self.out_proj = tf.keras.layers.Dense(att_dim, use_bias=use_bias)

        self.head_dim = att_dim // att_head

        self.left_mask = left_mask
        self.right_mask = right_mask

        self.dropout = dropout
        self.renormalize_dropout = renormalize_dropout
        self.dropout_window = dropout_window
        self.dropout_column = dropout_column

    def call(self,
             x,
             x_len,
             training=True,
             stream=False,
             past_state=None,
             valid_state=None):
        """
        x: [batch, seq, H]
        i: query ([0, seq-1])
        j: key ([0, seq-1])
                   /*   query   */      /*  key  */
        A_i,j = (E_x_i' * W_q' + u') * W_k,E * E_x_j  /* content energy */
        """
        B, S, _ = shape_list(x)  # S = Q = K

        if stream:
            if self.left_mask == -1 or self.right_mask != 0:
                raise NotImplementedError('This layer cannot be streamed')
            # past_state: Tuple of (key, value).
            past_key, past_value = tf.split(past_state,
                                            [self.left_mask, self.left_mask],
                                            axis=1)

        # query
        q_con = self.content_query(x)  # E_x_i' * W_q' + u'
        q_con = tf.reshape(q_con,
                           [B, S, self.num_head, self.head_dim])  # [B,Q,N,D]

        k_con = self.content_key(x)  # W_k,E * E_x_j
        k_con = tf.reshape(k_con,
                           [B, S, self.num_head, self.head_dim])  # [B,Q,N,D]

        if stream:
            k_con = tf.concat([past_key, k_con], axis=1)

        if q_con.shape[0] is not None:
            content_energy = einsum(q_con,
                                    k_con,
                                    trans1=[0, 2, 1, 3],
                                    trans2=[0, 2, 3, 1])
        else:
            content_energy = tf.einsum('bind,bjnd->bnij', q_con,
                                       k_con)  # [B,N,Q,K]

        energy = tf.divide(content_energy,
                           tf.math.sqrt(
                               tf.cast(self.head_dim,
                                       content_energy.dtype)))  # [B,N,Q,K]

        dropout = self.dropout if training else 0

        attention = compute_weight_from_energy(
            energy,
            x_len,
            self.left_mask,
            self.right_mask,
            stream=stream,
            valid_state=valid_state,
            dropout=dropout,
            renormalize_dropout=self.renormalize_dropout,
            dropout_window=self.dropout_window,
            dropout_column=self.dropout_column)

        v = tf.reshape(self.value(x),
                       [B, S, self.num_head, self.head_dim])  # [B,K,N,D]
        if stream:
            v = tf.concat([past_value, v], axis=1)

        context = compute_context(attention, v, stream=stream)  # [B, Q, N*D]

        if self.use_proj:
            context = tf.reshape(context, [B, S, x.shape[2]])
            context = self.out_proj(context)

        if stream:
            if x_len is None:
                next_key = k_con[:, -self.left_mask:, :, :]
                next_value = v[:, -self.left_mask:, :, :]
            else:
                next_key = layer_utils.get_last_values(k_con, x_len,
                                                       self.left_mask)
                next_value = layer_utils.get_last_values(
                    v, x_len, self.left_mask)

            next_state = tf.concat([next_key, next_value], axis=1)
            return context, next_state

        return context

    def initial_state(self, batch_size):
        # initial state for self-attention is empty
        state_size = [batch_size, self.left_mask, self.num_head, self.head_dim]
        return tf.concat([tf.zeros(state_size), tf.zeros(state_size)], axis=1)

    def stream(self,
               inputs,
               state,
               seq_len=None,
               valid_state=None,
               training=False):
        #TODO: remove stream() and use call() directly
        return self.call(x=inputs,
                         x_len=seq_len,
                         stream=True,
                         past_state=state,
                         valid_state=valid_state)


class AddAttention(tf.keras.layers.Layer):
    """ Additive Attention Layer (a.k.a Bahdanau Attention)
    I referred https://arxiv.org/abs/1409.0473.
    Currently, single-head is supported. Multi-head will be added soon.

    Args:
        att_dim: dimension of attention
        num_att_head: number of attention head
    """

    def __init__(self, att_dim, num_att_head):
        assert num_att_head == 1, "Additive attention only supports single head"
        super(AddAttention, self).__init__()
        self.att_dim = att_dim

        self.query_trans = tf.keras.layers.Dense(self.att_dim)
        self.key_trans = tf.keras.layers.Dense(self.att_dim)
        self.att = tf.keras.layers.AdditiveAttention(use_scale=True)

    def call(self,
             query,
             value,
             value_lens,
             key_adv=None,
             value_adv=None,
             transpose_value=False):
        if transpose_value:
            raise NotImplementedError("transpose_value is not supported")

        if value is None:
            assert key_adv is not None
            assert value_adv is not None
            key = key_adv
            value = value_adv
        else:
            assert key_adv is None
            assert value_adv is None
            key = self.key_trans(value)

        query = tf.expand_dims(self.query_trans(query), axis=1)
        if value_lens is None:
            mask = None
        else:
            mask = [None, tf.sequence_mask(value_lens)]
        context = self.att(inputs=[query, value, key], mask=mask)
        return context


class DotAttention(tf.keras.layers.Layer):

    def __init__(self, att_dim, num_att_head=1):
        """ Dot Attention Layer (with multi-headed method)
        You can find equations section 3.2, https://arxiv.org/pdf/1706.03762.pdf

        Args:
            att_dim: dimension of attention
            num_att_head: number of attention head

        Inputs:
            query: [B, T_q, dim] or [B, dim]
            value: [B, T_v, dim]
            value_lens: [B]

        """
        super(DotAttention, self).__init__()
        self.att_dim = att_dim
        self.num_att_head = num_att_head

        self.query_trans = tf.keras.layers.Dense(self.att_dim)
        self.key_trans = tf.keras.layers.Dense(self.att_dim)
        self.value_trans = tf.keras.layers.Dense(self.att_dim)
        if self.num_att_head > 1:
            self.final_att = tf.keras.layers.Dense(self.att_dim)

    def call(self,
             query,
             value,
             value_lens,
             key_adv=None,
             value_adv=None,
             transpose_value=False):
        if transpose_value:
            assert value is None
            att_axis = -2
            seq_axis = -1
        else:
            att_axis = -1
            seq_axis = -2

        if value is None:
            assert key_adv is not None
            assert value_adv is not None
            key = key_adv
            value = value_adv
        else:
            assert key_adv is None
            assert value_adv is None
            key = self.key_trans(value)  # [B, T_enc, D]
            value = self.value_trans(value)  # [B, T_enc, D]

        # query [B, D] or [B, T, D]
        if len(query.shape) == 2:
            # if query [B, D], reshape to [B, T_dec=1, D]
            query = tf.expand_dims(query, axis=1)

        att_dim_per_head = self.att_dim // self.num_att_head
        query = self.query_trans(query)  # [B, T_dec, D]
        query = tf.divide(query,
                          tf.math.sqrt(tf.cast(att_dim_per_head, query.dtype)))

        key_shape = shape_list(key)
        query_shape = shape_list(query)
        value_shape = shape_list(value)
        key = tf.reshape(key,
                         [*key_shape[:2], self.num_att_head, att_dim_per_head
                          ])  # [BK,K,N,D]
        query = tf.reshape(
            query, [*query_shape[:2], self.num_att_head, att_dim_per_head
                    ])  # [BQ,Q,N,D]
        value_shape[att_axis] = att_dim_per_head
        value_shape.insert(att_axis, self.num_att_head)
        value = tf.reshape(value, value_shape)  # [BK,S,N,D] or [BK,N,S,D]

        energy_mask_shape = [key_shape[0], 1, 1, key_shape[1]]  # [BK,1,1,K]
        if key.shape[0] == 1 and query.shape[1] == 1:
            # encoder's batch and decoder's timestep are both 1
            key = tf.squeeze(key, axis=0)  # [K,N,D]
            query = tf.squeeze(query, axis=1)  # [BQ,N,D]
            energy = einsum(key,
                            query,
                            trans1=[1, 0, 2],
                            trans2=[1, 0, 2],
                            trans3=[0, 2, 1],
                            transpose_b=True)  # [N,BQ,K]
            energy_mask_shape = [1, 1, key_shape[1]]  # [1,BK=1,K]
        elif query.shape[0] is not None:
            energy = einsum(query,
                            key,
                            trans1=[0, 2, 1, 3],
                            trans2=[0, 2, 3, 1])
        else:
            energy = tf.einsum('bind,bjnd->bnij', query, key)  # [B,N,Q,K]

        if value_lens is None:
            mask = None
        else:
            if len(value_lens.shape) == 2:
                mask = value_lens
            else:
                mask = tf.sequence_mask(value_lens,
                                        maxlen=key_shape[1],
                                        dtype=query.dtype)
            mask = tf.reshape(mask, energy_mask_shape)

        att_weight = masked_softmax(energy, mask)

        att = compute_context(att_weight,
                              value,
                              transpose_value=transpose_value)

        if self.num_att_head > 1:
            context = self.final_att(att)
        else:
            context = att
        return context
