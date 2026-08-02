import tensorflow as tf

from speech.trainer.tf_based_end_to_end_trainer.tf_trainer.layers \
        import layer_utils


class UtilTest(tf.test.TestCase):

    def test_last_values(self):
        T = tf.constant([[[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0]],
                         [[4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [9, 1]],
                         [[7, 2], [8, 2], [9, 2], [10, 2], [11, 2], [12, 2]]])
        seq_len = tf.constant([3, 4, 5])
        output = layer_utils.get_last_values(T, seq_len, 2)

        expected_output = tf.constant([[[2, 0], [3, 0]], [[6, 1], [7, 1]],
                                       [[10, 2], [11, 2]]])

        self.assertAllClose(output, expected_output)

    def test_flatten_nested_tensors(self):
        flatten = layer_utils.flatten_nested_tensors

        # empty input test
        self.assertAllEqual([], flatten([]))

        T = tf.constant

        def _get_nested_input():
            return [T(1), [T(2), (T(3), T(4))], (T(5), T(6))]

        def _get_expected_output():
            return [T(1), T(2), T(3), T(4), T(5), T(6)]

        # flattened input test
        self.assertAllEqual(_get_expected_output(),
                            flatten(_get_expected_output()))

        # nested input test
        self.assertAllEqual(_get_expected_output(),
                            flatten(_get_nested_input()))

    def test_grad_scale(self):
        random_input_size = (3, 4)
        x = tf.random.uniform(random_input_size)

        with tf.GradientTape(persistent=True) as g:
            g.watch(x)

            y0 = layer_utils.grad_scale(x, 0.5)
            y1 = layer_utils.grad_scale(x, 2.0)
            y2 = tf.identity(x)

            l0 = tf.norm(y0)
            l1 = tf.norm(y1)
            l2 = tf.norm(y2)

        self.assertAllEqual(y0, y1)
        self.assertAllEqual(y1, y2)

        g0 = g.gradient(l0, x)
        g1 = g.gradient(l1, x)
        g2 = g.gradient(l2, x)

        self.assertAllClose(g0, 0.5 * g2)
        self.assertAllClose(g1, 2.0 * g2)


if __name__ == '__main__':
    tf.test.main()
