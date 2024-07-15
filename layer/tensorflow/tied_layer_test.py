import tensorflow as tf
import numpy as np

from speech.trainer.tf_based_end_to_end_trainer.tf_trainer.layers.tied_layer \
    import TiedDense

import unittest

class LayerTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(LayerTest, self).__init__(*args, **kwargs)
        self.input_size = 32
        self.hidden_size = 64

    def test_weight_share(self):
        ''' Test for weight sharing
        '''

        class Sample_model(tf.keras.Model):
            def __init__(self, input_size, hidden_size):
                super(Sample_model, self).__init__()
                self.emb = tf.keras.layers.Embedding(input_size, hidden_size)
                self.tied = TiedDense(input_size, tied_to=self.emb)

            def call(self, x):
                h = self.emb(x)
                p = self.tied(h)
                return p

        m = Sample_model(self.input_size, self.hidden_size)
        m.build(tf.TensorShape([1, self.input_size]))

        np.testing.assert_array_almost_equal(m.emb.embeddings.numpy(), m.tied.tied_kernel.numpy())

        x = tf.random.uniform([1, self.input_size])
        y = tf.random.uniform([1, self.input_size])
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

        with tf.GradientTape() as g:
            g.watch(x)
            p = m(x)
            loss = tf.norm(p - y)
        grad = g.gradient(loss, m.trainable_variables)
        optimizer.apply_gradients(zip(grad, m.trainable_variables))

        np.testing.assert_array_almost_equal(m.emb.embeddings.numpy(), m.tied.tied_kernel.numpy())

    def test_transpose_false(self):
        ''' Test for weight sharing
        '''

        class Sample_model(tf.keras.Model):
            def __init__(self, input_size, hidden_size):
                super(Sample_model, self).__init__()
                self.d1 = tf.keras.layers.Dense(hidden_size,
                                                activation='tanh')
                self.d2 = TiedDense(hidden_size,
                                    tied_to=self.d1,
                                    transpose_kernel=False,
                                    activation='relu')
                self.out = tf.keras.layers.Dense(input_size)

            def call(self, x):
                h1 = self.d1(x)
                h2 = self.d2(x)
                o = self.out(h1 + h2)
                return o

        m = Sample_model(self.input_size, self.hidden_size)
        m.build(tf.TensorShape([1, self.input_size]))

        np.testing.assert_array_almost_equal(m.d1.kernel.numpy(), m.d2.tied_kernel.numpy())
        np.testing.assert_array_almost_equal(m.d1.bias.numpy(), m.d2.bias.numpy())

        x = tf.random.uniform([1, self.input_size])
        y = tf.random.uniform([1, self.input_size])
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

        with tf.GradientTape() as g:
            g.watch(x)
            p = m(x)
            loss = tf.norm(p - y)
        grad = g.gradient(loss, m.trainable_variables)
        optimizer.apply_gradients(zip(grad, m.trainable_variables))

        np.testing.assert_array_almost_equal(m.d1.kernel.numpy(), m.d2.tied_kernel.numpy())
        np.testing.assert_raises(AssertionError, np.testing.assert_array_almost_equal,
                                 m.d1.bias.numpy(), m.d2.bias.numpy())

    def test_dense_op(self):
        dense = tf.keras.layers.Dense(self.hidden_size,
                                      activation='tanh')
        tied_dense = TiedDense(self.hidden_size,
                               tied_to=dense,
                               transpose_kernel=False,
                               activation='tanh')

        x = tf.random.uniform([1, self.input_size])
        np.testing.assert_array_almost_equal(dense(x), tied_dense(x))


    def test_grad_flow(self):
        ''' Test for gradients from tied layer
        '''

        class Sample_model(tf.keras.Model):
            def __init__(self, input_size, hidden_size):
                super(Sample_model, self).__init__()
                self.emb = tf.keras.layers.Embedding(input_size, hidden_size)
                self.tied = TiedDense(input_size, tied_to=self.emb)

            def call(self, x):
                h = self.emb(x)
                h = tf.stop_gradient(h)
                p = self.tied(h)
                return p

        m = Sample_model(self.input_size, self.hidden_size)
        m.build(tf.TensorShape([1, self.input_size]))

        np.testing.assert_array_almost_equal(m.emb.embeddings.numpy(), m.tied.tied_kernel.numpy())

        old_embedding = m.emb.embeddings.numpy()

        x = tf.random.uniform([1, self.input_size])
        y = tf.random.uniform([1, self.input_size])
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

        with tf.GradientTape() as g:
            g.watch(x)
            p = m(x)
            loss = tf.norm(p - y)
        grad = g.gradient(loss, m.trainable_variables)
        optimizer.apply_gradients(zip(grad, m.trainable_variables))

        np.testing.assert_array_almost_equal(m.emb.embeddings.numpy(), m.tied.tied_kernel.numpy())
        np.testing.assert_raises(AssertionError, np.testing.assert_array_almost_equal, old_embedding, m.emb.embeddings.numpy())

if __name__ == '__main__':
    unittest.main()
