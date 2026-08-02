#!/usr/bin/python3

# pylint: disable=invalid-name, protected-access, import-error
# pylint: disable=no-name-in-module

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

import os
import tensorflow as tf

from speech.layers import masking_layer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from speech.trainer.util.tfdeterminism import set_global_determinism

set_global_determinism()


def create_simple_model():
    inputs = tf.keras.layers.Input(shape=(3, 2))
    masking = (masking_layer.Masking(
        masking_layer.MaskingType.SMALL_VALUE_MASKING, dynamic=False)(inputs))
    return tf.keras.models.Model(inputs, outputs=masking)


# Understands why it cannot capture the batch size information.

inputs = tf.reshape(tf.range(24), (4, 3, 2))
print(inputs)
model = create_simple_model()

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,
                                                            reduction="none")

model.compile(loss=loss_object,
              optimizer=optimizer,
              metrics=["accuracy"],
              run_eagerly=True)

print("++++++++++++++++")

outputs = model(inputs, training=True)

print("=========")
print(outputs)
