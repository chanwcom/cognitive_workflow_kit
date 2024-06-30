# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import glob
import os

# Third-party imports
from transformers import AutoModelForCTC, TrainingArguments, Trainer
from datasets import load_dataset, Audio
from transformers import AutoProcessor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from torch.utils import data
import tensorflow as tf
import torch
import evaluate
import numpy as np

# Custom imports
from data.format import speech_data_helper
from typing import Any, Dict, List, Optional, Union
from loss.pytorch import seq_loss_util

# Prevents Tensorflow from using the entire GPU memory.
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

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

class IterDataset(data.IterableDataset):
    def __init__(self, tf_dataset):
        self._dataset = tf_dataset

    def __iter__(self):
        for data in self._dataset:
            output = {}
            output["input_values"] = [tf.squeeze(data[0]["SEQ_DATA"]).numpy()]
            output["input_length"] = tf.squeeze(data[0]["SEQ_LEN"]).numpy()
            with processor.as_target_processor():
                output["labels"] = processor(
                    data[1]["SEQ_DATA"][0].numpy().decode(
                        "unicode_escape")).input_ids

            yield (output)


pytorch_train_dataset = IterDataset(train_dataset)
pytorch_test_dataset = IterDataset(test_dataset)


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = evaluate.load("wer")
    wer = wer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{
            "input_values": feature["input_values"][0]
        } for feature in features]
        label_features = [{
            "input_ids": feature["labels"]
        } for feature in features]

        batch = self.processor.pad(input_features,
                                   padding=self.padding,
                                   return_tensors="pt")

        labels_batch = self.processor.pad(labels=label_features,
                                          padding=self.padding,
                                          return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor,
                                           padding="longest")

model = AutoModelForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id)

training_args = TrainingArguments(
    output_dir=
    "/home/chanwcom/local_repositories/cognitive_workflow_kit/tool/models/asr_stop_model_final3",
    per_device_train_batch_size=40,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=1000,
    max_steps=10000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=40,
    save_steps=5000,
    eval_steps=100,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        target = inputs.pop("labels")
        outputs = model(**inputs)

        # In torch.nn.CTCLoss(), the logit should have the shape of (T, B, C).
        logits = torch.permute(outputs["logits"], (1, 0, 2))
        logits_lengths = torch.full(size=(logits.shape[1],), fill_value=logits.shape[0])
        target_lengths = torch.sum((target >= 0).type(torch.int32), axis=1)

        ctc_loss = torch.nn.CTCLoss()

        loss = ctc_loss(logits.log_softmax(2), target, logits_lengths, target_lengths)

        if return_outputs:
            return loss, outputs
        else:
            return loss

class MyCtcTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        blank_augmented_inputs = {}
        blank_augmented_inputs["SEQ_DATA"] = inputs["labels"]
        blank_augmented_inputs["SEQ_LEN"] = torch.sum(
            (inputs["labels"] >= 0).type(torch.int32), axis=1)

        with torch.device(inputs["input_values"].device.type):
            blank_augmented_inputs = seq_loss_util.to_blank_augmented_labels(
                blank_augmented_inputs, 0, False)

        target = inputs.pop("labels")
        outputs = model(**inputs)

        logits = outputs["logits"]
        logits_lengths = torch.full(size=(logits.shape[1],), fill_value=logits.shape[0])

        target = blank_augmented_inputs["SEQ_DATA"]
        target_lengths = blank_augmented_inputs["SEQ_LEN"]

        with torch.device(inputs["input_values"].device.type):
            ctc_loss = seq_loss_util.CtcLoss()
            loss = ctc_loss.apply(
                target, target_lengths, logits.log_softmax(2), logits_lengths)

        if return_outputs:
            return loss, outputs
        else:
            return loss

trainer = MyCtcTrainer(
    model=model,
    args=training_args,
    train_dataset=pytorch_train_dataset,
    eval_dataset=pytorch_test_dataset,
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
