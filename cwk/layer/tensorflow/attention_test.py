#!/usr/bin/env python

# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

import numpy as np
import tensorflow as tf

from speech.trainer.tf_based_end_to_end_trainer.tf_trainer.layers.attention \
        import RelativeAttention, SelfAttention
from speech.trainer.tf_based_end_to_end_trainer.tf_trainer.layers.attention \
        import RelativePositionEncoding
from speech.trainer.tf_based_end_to_end_trainer.tf_trainer.layers.attention \
        import sequence_mask

# test can fail with TF32 format
from packaging import version
if version.parse(tf.__version__) >= version.parse("2.4"):
    tf.config.experimental.enable_tensor_float_32_execution(False)


class SelfAttentionTest(tf.test.TestCase):

    class SampleModel(tf.keras.Model):

        def __init__(self, att_dim, num_head, left_mask, right_mask):
            super().__init__()
            self.att = SelfAttention(att_dim,
                                     num_head,
                                     left_mask=left_mask,
                                     right_mask=right_mask)

        def call(self, x, x_len):
            return self.att(x, x_len)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_dim = 4
        self.num_head = 1
        self.seq_len = 12
        self.hidden_dim = self.head_dim * self.num_head
        self.left_mask = 2
        self.right_mask = 0

    def test_mask(self):
        # test for masking and seq_len
        m = self.SampleModel(self.head_dim * self.num_head, self.num_head,
                             self.left_mask, self.right_mask)

        x = tf.random.uniform([3, self.seq_len, self.hidden_dim])
        y = tf.random.uniform([3, self.seq_len, self.head_dim * self.num_head])

        with tf.GradientTape() as tape:
            out = m(x, [5, 4, 3])
            loss = tf.norm(out - y)

        grad = tape.gradient(loss, m.trainable_variables)

        for g, v in zip(grad, m.trainable_variables):
            assert not np.isnan(g.numpy()).any()

        tf.tensor_scatter_nd_update(x, [[1, 4]],
                                    tf.random.uniform([1, self.hidden_dim]))
        tf.tensor_scatter_nd_update(x, [[2, 3]],
                                    tf.random.uniform([1, self.hidden_dim]))
        tf.tensor_scatter_nd_update(x, [[2, 4]],
                                    tf.random.uniform([1, self.hidden_dim]))

        with tf.GradientTape() as tape:
            out2 = m(x, [5, 4, 3])
            loss = tf.norm(out - y)

        self.assertAllClose(out[:, :3, :], out2[:, :3, :])

    def test_mask_fp16(self):
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        m = self.SampleModel(self.head_dim * self.num_head, self.num_head,
                             self.left_mask, self.right_mask)

        x = tf.random.uniform([3, self.seq_len, self.hidden_dim])
        y = tf.random.uniform([3, self.seq_len, self.head_dim * self.num_head])

        with tf.GradientTape() as tape:
            out = m(x, [5, 4, 3])
            loss = tf.norm(out - tf.cast(y, tf.float16))

        grad = tape.gradient(loss, m.trainable_variables)

        for g, v in zip(grad, m.trainable_variables):
            assert not np.isnan(g.numpy()).any()

        policy = mixed_precision.Policy('float32')
        mixed_precision.set_policy(policy)

    def test_stream(self):
        m = self.SampleModel(self.head_dim * self.num_head, self.num_head,
                             self.left_mask, self.right_mask)
        layer = m.att
        batch_size = 3

        initial_state = layer.initial_state(batch_size)
        test_input = tf.random.uniform([batch_size, self.seq_len, 3])

        # Nonstreaming inference
        nonstream_output = layer(test_input, x_len=[self.seq_len])
        # Streaming inference
        chunk_size = 3
        _state = initial_state
        stream_output = []
        for i in range(0, test_input.shape[1], chunk_size):
            _input = test_input[:, i:i + chunk_size, :]
            _output, _state = layer.stream(_input, _state)
            stream_output.append(_output)
        stream_output = tf.concat(stream_output, axis=1)

        # NOTE: The first outputs are different because of
        #       zero key/values in the initial state.
        self.assertAllClose(nonstream_output[:, self.left_mask:, :],
                            stream_output[:, self.left_mask:, :])


class RelativeAttentionTest(tf.test.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_dim = 3
        self.seq_len = 5
        self.hidden_dim = 4

    def test_gradient(self):

        class SampleModel(tf.keras.Model):

            def __init__(self, att_dim, num_head, hidden_dim):
                super().__init__()
                self.att = RelativeAttention(att_dim, num_head, hidden_dim)

            @tf.function
            def call(self, x, x_len):
                return self.att(x, x_len)

        def _test_with_num_head(num_head):
            m = SampleModel(self.head_dim * num_head, num_head,
                            self.hidden_dim)

            x = tf.random.uniform([1, self.seq_len, self.hidden_dim])
            y = tf.random.uniform([1, self.seq_len, self.head_dim * num_head])

            with tf.GradientTape() as tape:
                out = m(x, [self.seq_len])
                loss = tf.norm(out - y)

            grad = tape.gradient(loss, m.trainable_variables)

            for g, v in zip(grad, m.trainable_variables):
                assert not np.isnan(g.numpy()).any()

        _test_with_num_head(1)
        _test_with_num_head(4)

    def test_rel_shift(self):
        x = tf.range(-self.seq_len + 1, self.seq_len, dtype=tf.float32)
        x = tf.tile(tf.expand_dims(x, axis=0), [self.seq_len, 1])
        x = tf.expand_dims(tf.expand_dims(x, axis=0), axis=0)
        self.assertAllClose(
            RelativeAttention.rel_shift(x),
            [[[[0, 1, 2, 3, 4], [-1, 0, 1, 2, 3], [-2, -1, 0, 1, 2],
               [-3, -2, -1, 0, 1], [-4, -3, -2, -1, 0]]]])

    def get_relative_energy_correct(self, q_rel, k_rel):
        shape = tf.shape(k_rel)  # [2*K-1, N, D]
        seq_len = tf.shape(q_rel)[1]  # Q (= K)
        idx = tf.range(seq_len)

        def fn(t):
            k_rel_t = tf.slice(k_rel, [seq_len - t - 1, 0, 0],
                               [seq_len, shape[1], shape[2]])  # [K, N, D]
            q_rel_t = q_rel[:, t, :, :]
            qk_rel = tf.einsum('bnd,jnd->bnj', q_rel_t, k_rel_t)  # [B, N, K]
            return qk_rel

        x = tf.map_fn(fn, idx, fn_output_signature=tf.float32)  # [Q, B, N, K]
        x = tf.transpose(x, [1, 2, 0, 3])  # [B, N, Q, K]
        return x

    def test_relshift_energy(self):
        import time
        num_head = 4
        batch_size = 128
        self.seq_len = 512
        self.head_dim = 1536
        k_rel = tf.random.uniform(
            [2 * self.seq_len - 1, num_head, self.head_dim])
        q_rel = tf.random.uniform(
            [batch_size, self.seq_len, num_head, self.head_dim])

        start = time.perf_counter()
        expected_energy = self.get_relative_energy_correct(q_rel, k_rel)
        print('map', time.perf_counter() - start)

        start = time.perf_counter()
        energy = tf.einsum('bind,jnd->bnij', q_rel, k_rel)
        energy = RelativeAttention.rel_shift(energy)
        print('shift', time.perf_counter() - start)

        self.assertAllClose(expected_energy, energy)


class UtilTest(tf.test.TestCase):

    def test_sequence_mask(self):
        seq_len = 8
        max_seq = 16
        lengths = tf.random.uniform([seq_len],
                                    minval=1,
                                    maxval=max_seq,
                                    dtype=tf.int32)
        self.assertAllClose(
            sequence_mask(lengths, maxlen=max_seq, dtype=tf.float16),
            tf.sequence_mask(lengths, maxlen=max_seq, dtype=tf.float16))


if __name__ == '__main__':
    tf.test.main()
