class BatchProbDropout(tf.keras.layers.Layer, AbstractDropout):

    def __init__(self, params_proto: dropout_params_pb2.DropoutParams,
                 **kwargs):
        super(BatchProbDropout, self).__init__(**kwargs)
        self._maintain_pattern = params_proto.maintain_pattern

    def call(self, inputs, drop_prob, training=None):
        """Runs the UniformDropout layer.

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
