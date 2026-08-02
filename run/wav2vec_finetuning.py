# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import os

# Third-party imports
from transformers import AutoModelForCTC, TrainingArguments, Trainer
from transformers import AutoProcessor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch
import evaluate
import numpy as np

# Custom imports
import sample_util

db_top_dir = "/mnt/data/database"
train_top_dir = os.path.join(db_top_dir, "stop_music/music_train")
test_top_dir = os.path.join(db_top_dir, "stop_music/music_test0")
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
output_dir = "/mnt/data/home/chanwcom/experiment/wav2vec2_stop_model_final"

train_dataset = sample_util.make_dataset(train_top_dir)
test_dataset = sample_util.make_dataset(test_top_dir)


def compute_metrics(pred) -> Dict[str, float]:
    """Compute word error rate (WER) between predictions and labels.

    This function decodes the model's predicted token IDs and ground truth
    label IDs into strings, replacing ignored label tokens with the padding
    token ID. Then it computes WER using the `evaluate` library.

    Args:
        pred: A prediction object with attributes:
            - predictions: logits or probabilities of shape
                (batch_size, seq_len, vocab_size).
            - label_ids: ground truth token IDs with padding replaced by -100.

    Returns:
        Dict[str, float]: Dictionary with WER under the key 'wer'.
    """
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Replace -100 in labels with tokenizer pad token ID to enable decoding
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer_metric = evaluate.load("wer")
    wer_score = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer_score}


@dataclass
class DataCollatorCTCWithPadding:
    """Data collator that dynamically pads input values and labels for CTC training.

    This class pads the input audio features and the corresponding label sequences
    (token IDs) to the length of the longest element in the batch. It also replaces
    padding tokens in the labels with -100 to ensure they are ignored during the loss
    computation, as required by PyTorch's CTC loss implementation.

    Attributes:
        processor (AutoProcessor): The processor used for feature extraction and tokenization.
        padding (Union[bool, str]): Padding strategy. Defaults to "longest" to pad to the
            longest sequence in the batch.
    """

    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """Pad inputs and labels in a batch for model training.

        Args:
            features: A list of feature dictionaries, each containing:
                - "input_values": the audio features (list or tensor).
                - "labels": the tokenized label sequence.

        Returns:
            A dictionary with padded input tensors and labels ready for the model:
            - "input_values": Padded input audio feature tensor.
            - "labels": Padded label tensor with padding tokens replaced by -100.
        """
        # Separates the input audio features and label sequences from the batch.
        input_features = [{"input_values": feature["input_values"]} for feature in features]

        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Uses the processor's pad method to pad input audio features to the same length.
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )

        # Pads the label sequences separately using the processor's pad method.
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt"
        )

        # Replaces padding tokens in labels with -100 so that the loss function ignores them.
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Add the processed labels to the batch dictionary.
        batch["labels"] = labels

        return batch

# Instantiate the data collator for CTC loss with padding support.
# It dynamically pads the inputs and labels in each batch to the longest
# sequence, enabling efficient batch processing without manual padding.
data_collator = DataCollatorCTCWithPadding(
    processor=processor,
    padding="longest"
)

# Load the pretrained Wav2Vec2 model with CTC (Connectionist Temporal Classification)
# head for speech recognition.
# - ctc_loss_reduction="mean" averages the CTC loss over the batch.
# - pad_token_id is set to the tokenizer's pad token to ensure correct masking.
model = AutoModelForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id
)

# Define the training arguments for the Hugging Face Trainer.
# These control training hyperparameters and runtime behavior:
training_args = TrainingArguments(
    # Directory to save model checkpoints and outputs.
    output_dir=output_dir,

    # Batch size per device (GPU/CPU) for training.
    per_device_train_batch_size=40,

    # Number of batches to accumulate gradients over before updating model weights.
    gradient_accumulation_steps=2,

    # Initial learning rate for the optimizer.
    learning_rate=1e-4,

    # Number of warmup steps to gradually increase learning rate at start.
    warmup_steps=1000,

    # Total number of training steps.
    max_steps=2000,

    # Enable gradient checkpointing to reduce memory usage at the cost of extra compute.
    gradient_checkpointing=True,

    # Use mixed precision training (float16) to speed up training and reduce memory.
    fp16=True,

    # Perform evaluation every N steps (eval_strategy="steps").
    eval_strategy="steps",

    # Batch size per device during evaluation.
    per_device_eval_batch_size=40,

    # Save model checkpoints every N steps.
    save_steps=2000,

    # Run evaluation every N steps during training.
    eval_steps=100,

    # Log training progress every N steps.
    logging_steps=25,

    # Load the best model (lowest WER) at the end of training automatically.
    load_best_model_at_end=True,

    # Metric to use for selecting the best model checkpoint.
    metric_for_best_model="wer",

    # Indicates that a lower metric score (WER) is better.
    greater_is_better=False,

    # Disable pushing model to the Hugging Face hub.
    push_to_hub=False,
)

# Create the Trainer instance to handle training and evaluation.
# This ties together the model, datasets, tokenizer, data collator, and metrics.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
