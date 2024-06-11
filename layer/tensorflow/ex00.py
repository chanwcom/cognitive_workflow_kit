#!/usr/bin/python3

import tensorflow as tf

counts = [1, 1]
# Probability of success.
probs = [0.8]

outputs = tf.random.uniform([1], 0.0, 1.0, tf.dtypes.float32)

print(outputs)

inputs = tf.reshape(tf.random.uniform(shape=[24], dtype=tf.dtypes.float32),
                    (4, 6))

print(inputs)

dropped = tf.nn.dropout(x=inputs, rate=outputs[0])

print(dropped)
