class UniformDropout(tf.keras.layers.Layer, AbstractDropout):
    """A keras layer implementation of UniformDropout layer.

    The dropout out probability is selected by a uniform distribution between
    drop_prob_min and drop_prob_max. If both drop_prob_min and drop_prob_max
    are the same, this layer is exactly the same as the normal dropout.
    """
    def __init__(self,
                 drop_prob_min=0.0,
                 drop_prob_max=0.0,
                 maintain_pattern=True,
                 **kwargs) -> None:
        """Initializes a UniformDropout layer.

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
        super(UniformDropout, self).__init__(**kwargs)

        assert drop_prob_min <= drop_prob_max, (
            "drop_prob_min should be equal to or less than drop_prob_max.")
        assert drop_prob_min >= 0.0 and drop_prob_max <= 1.0, (
            "The prob_min and prob_max must be in the interval [0.0, 1.0].")

        self._drop_prob_min = drop_prob_min
        self._drop_prob_max = drop_prob_max
        self._maintain_pattern = maintain_pattern

    def call(self, inputs, training=None):
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

        if self._drop_prob_min == self._drop_prob_max:
            drop_prob = self._drop_prob_min * tf.ones(shape=batch_size)
        elif self._drop_prob_min < self._drop_prob_max:
            # Selects a random probability between self._drop_prob_min and
            # self._drop_prob_max.
            drop_prob = tf.random.uniform([batch_size],
                                          self._drop_prob_min,
                                          self._drop_prob_max,
                                          dtype=inputs.dtype)
        else:
            raise ValueError(
                "drop_prob_min should be equal to or less than drop_prob_max.")

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
