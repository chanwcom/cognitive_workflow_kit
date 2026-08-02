# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import argparse
import os

# Third-party imports
import torch
import evaluate
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers import AutoModelForCTC, TrainingArguments, Trainer
from transformers import AutoProcessor

# Custom imports
from run import sample_util
from loss.pytorch import seq_loss_util

# Global directory settings
db_top_dir = "/mnt/data/database"
train_top_dir = os.path.join(db_top_dir, "libri_light/1h")
test_top_dir = os.path.join(
    db_top_dir, "libri_speech_webdataset_new_oct_2025/test-clean")
model_top_dir = "/mnt/data/home/chanwcom/models"
spm_top_dir = ("/mnt/data/home/chanwcom/local_repository/"
               "cognitive_workflow_kit_work/run/resources")

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")


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
    """Data collator that dynamically pads inputs and labels for CTC training.

    This class pads the input audio features and the corresponding label
    sequences to the length of the longest element in the batch. It also
    replaces padding tokens in the labels with -100.

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
        input_features = [{"input_values": feature["input_values"]}
                          for feature in features]
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]
        # Use the processor's pad method to pad input audio features to the same length.
        # Without return_attention_mask, Wav2Vec2 does not generate the mask.
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
            return_attention_mask=True
        )

        # Pad the label sequences separately using the processor's pad method.
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt"
        )

        # Replace padding tokens in labels with -100
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Add the processed labels to the batch dictionary.
        batch["labels"] = labels
        return batch


class MyCtcTrainer(Trainer):
    """Custom Trainer to override loss computation with custom CTC loss."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        with torch.device(inputs["input_values"].device.type):
            target = inputs.pop("labels")
            outputs = model(**inputs)

            input_lengths = inputs["attention_mask"].sum(-1)
            logits_lengths = model._get_feat_extract_output_lengths(
                input_lengths)

            logits = outputs["logits"]
            target_lengths = torch.sum(
                (target >= 0).type(torch.int32), axis=1)

            ctc_loss = seq_loss_util.CtcLoss()

            # Custom CTC Loss implementation
            loss = ctc_loss.apply(
                target,
                target_lengths,
                logits.log_softmax(2),
                logits_lengths,
                seq_loss_util.LabelType.CTC,
                False,
                seq_loss_util.ThresholdType.LS,
                0.97,
            ).mean()

        if return_outputs:
            return loss, outputs
        else:
            return loss


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Wav2Vec2 Training with Dynamic Vocab Size")
    parser.add_argument("--vocab_size", type=int, default=None,
                        help="Vocabulary size (e.g., 32, 128).")
    return parser.parse_args()


def main():
    args = parse_args()

    # Dynamic configuration based on vocab_size
    if args.vocab_size is not None:
        spm_name = f"librispeech_unigram_{args.vocab_size}.model"
        spm_model_path = os.path.join(spm_top_dir, spm_name)
        current_vocab_size = args.vocab_size
        out_name = f"ls_0p97_2000_steps_unigram_{args.vocab_size}_03"
    else:
        spm_model_path = None
        current_vocab_size = 32
        out_name = "ls_0p97_2000_steps_default_vocab_03"

    # Dataset preparation
    train_dataset = sample_util.make_dataset(
        train_top_dir, True, spm_model_path)
    test_dataset = sample_util.make_dataset(
        test_top_dir, True, spm_model_path)

    # Initialize data collator
    data_collator = DataCollatorCTCWithPadding(
        processor=processor, padding="longest")

    # Load model with dynamic vocab size
    model = AutoModelForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=current_vocab_size,
        ignore_mismatched_sizes=True
    )

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join("/mnt/data/home/chanwcom/models", out_name),
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_steps=500,
        max_steps=2000,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=24,
        save_steps=2000,
        eval_steps=200,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    # Initialize trainer and start training
    trainer = MyCtcTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()
