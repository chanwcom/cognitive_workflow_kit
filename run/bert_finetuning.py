# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import glob
import os

# Third-party imports
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils import data
import tensorflow as tf
import torch
import evaluate
import numpy as np

# Custom imports
from data import speech_data_helper

# Preventing Tensorflow from using the entire GPU memory.
#
# Since we use Tensorflow and Pytorch simultaneously, Tensorflow shouldl not
# occupy the entire memory. Instead of allocating the entire GPU memory, GPU
# memory allocated to Tensorflow grows based on its need. Refer to the
# following website for more information:
# https://www.tensorflow.org/guide/gpu
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs.
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized.
        print(e)

db_top_dir = "/home/chanwcom/databases/"
train_top_dir = os.path.join(db_top_dir, "stop/music_train_tfrecord")
test_top_dir = os.path.join(db_top_dir,
                            "stop/test_0_music_random_300_tfrecord")

# yapf: disable
op = speech_data_helper.SpeechDataToWave()
train_dataset = tf.data.TFRecordDataset(
    glob.glob(os.path.join(train_top_dir, "*tfrecord-*")),
              compression_type="GZIP")
train_dataset = train_dataset.batch(1)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.map(
    op.process, num_parallel_calls=tf.data.AUTOTUNE)
# yapf: enable

# yapf: disable
test_dataset = tf.data.TFRecordDataset(
    glob.glob(os.path.join(test_top_dir, "*tfrecord-*")),
              compression_type="GZIP")
test_dataset = test_dataset.batch(1)
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.map(op.process)
# yapf: enable

model_name = "google-bert/bert-base-uncased"


def find_index(inputs):
    intents = [
        "ADD_TO_PLAYLIST_MUSIC",
        "CREATE_PLAYLIST_MUSIC",
        "DISLIKE_MUSIC",
        "LIKE_MUSIC",
        "LOOP_MUSIC",
        "PAUSE_MUSIC",
        "PLAY_MUSIC",
        "PREVIOUS_TRACK_MUSIC",
        "REMOVE_FROM_PLAYLIST_MUSIC",
        "REPLAY_MUSIC",
        "SKIP_TRACK_MUSIC",
        "START_SHUFFLE_MUSIC",
        "STOP_MUSIC",
        "UNSUPPORTED_MUSIC",
    ]

    index = -1
    for (i, intent) in enumerate(intents):
        if intent == inputs:
            index = i
            break

    assert index >= 0

    return index


model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                           num_labels=14)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(examples, padding="max_length", truncation=True)


metric = evaluate.load("accuracy")


class IterDataset(data.IterableDataset):

    def __init__(self, tf_dataset):
        self._dataset = tf_dataset
        op = speech_data_helper.SpeechDataToWave()

    def __iter__(self):
        for data in self._dataset:
            output = {}
            output["label"] = find_index(data[3][0][0])
            output["text"] = data[1]["SEQ_DATA"][0].numpy().decode("utf-8")
            tokenized = tokenize_function(output["text"])
            for key in tokenized.keys():
                output[key] = tokenized[key]

            yield (output)


pytorch_train_dataset = IterDataset(train_dataset)
pytorch_test_dataset = IterDataset(test_dataset)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir=
    "/home/chanwcom/local_repositories/cognitive_workflow_kit/tool/models/bert_model",
    per_device_train_batch_size=40,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=500,
    max_steps=1000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=40,
    save_steps=1000,
    eval_steps=100,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=pytorch_train_dataset,
    eval_dataset=pytorch_test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
