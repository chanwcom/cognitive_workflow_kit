# pylint: disable=import-error, no-member
"""Training script for the SHC-loss wav2vec2 CTC model.

===============================================================================
CLEANUP NOTES (what changed vs. the previous version, and why)
===============================================================================

1) `out_name` no longer hardcodes hyperparameter values into the run name.
   -----------------------------------------------------------------------
   The old code did:
       out_name = f"model_2000_steps_alpha_0p02_beta_0p0_unigram_{args.vocab_size}_00"
   This string is *always* "alpha_0p02_beta_0p0" and "2000_steps" no matter
   what `--alpha`/`--beta`/`--max_steps` you actually pass -- so two runs
   with different alpha/beta would silently get the same directory name
   (and one would overwrite the other's checkpoints). Two things fixed
   this:
     - `--run_name` is now a first-class CLI argument. If you pass it,
       that's the directory name, full stop -- no magic string-building.
     - If you don't pass `--run_name`, a name is still auto-generated for
       convenience, but from the *actual* `args.alpha` / `args.beta` /
       `args.max_steps` values (see `_default_run_name()`), so it's at
       least accurate. The generated name is also printed at startup so
       you always know what directory you're about to write to.

2) The `if 0: ... if 1: ...` GPU-profile switching pattern is replaced by
   `--gpu_profile {4090,a100}` plus per-flag overrides.
   -----------------------------------------------------------------------
   The old code had two full `TrainingArguments(...)` blocks (plus a third,
   fully commented-out legacy one) and you'd edit the source to flip which
   one was "if 1" vs "if 0" to switch machines. That's easy to get wrong
   (e.g. forgetting which one is active) and isn't scriptable. Now:
     - `_GPU_PROFILES` holds the two presets (their values are exactly the
       same numbers as the old "if 1" (4090) and "if 0" (A100) blocks).
     - `--gpu_profile` picks the base preset (default: "4090", matching
       the branch that was actually active before).
     - Any of the individual hyperparameter flags
       (`--per_device_train_batch_size`, `--learning_rate`, etc.) can be
       passed explicitly to override just that one value on top of the
       chosen profile -- no source editing needed.
   The old fully-commented-out legacy block (500 steps / 96 batch / 250
   warmup) was dropped; it wasn't reachable and was just clutter.

3) `processor` is no longer a mutated module-level global.
   -----------------------------------------------------------------------
   The old code declared `processor` at module scope and reassigned
   `processor.tokenizer` inside `main()` via `global processor`, and
   `compute_metrics()` read that same module-level `processor` implicitly.
   That's a bit fragile (any other code importing this module sees a
   half-initialized `processor` before `main()` runs, and it's not obvious
   from `compute_metrics`'s signature that it depends on outside state).
   `compute_metrics` is now built by `make_compute_metrics(processor)`, a
   factory that returns a closure capturing `processor` explicitly -- still
   satisfying the HF `Trainer` requirement that `compute_metrics` take a
   single `pred` argument, but without relying on a mutable global.

4) Removed dead code.
   -----------------------------------------------------------------------
   - `current_vocab_size` was assigned in both branches of the
     `vocab_size` if/else but never read afterwards. Removed.
   - The various commented-out `#torch.backends...` / `#os.environ...`
     lines and the big commented-out legacy `TrainingArguments` block were
     dropped. (If you need `CUDA_LAUNCH_BLOCKING=1` or TF32-disable for
     debugging, `--debug_sync` / `--disable_tf32` flags are provided
     instead of source comments to toggle.)
===============================================================================
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import argparse
import itertools
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

# Third-party imports
import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForCTC, AutoProcessor,
                          PreTrainedTokenizer, Trainer, TrainingArguments)

# Custom imports
from common import sample_util
from cwk.loss.pytorch import shc_loss

# Default directory settings. These are now just the *defaults* for the
# corresponding CLI flags (see parse_args()) rather than fixed module-level
# globals, so every path here can be overridden per-invocation without
# editing the source:
#   --db_top_dir           (was: db_top_dir)
#   --train_top_dir        (was: train_top_dir, derived from db_top_dir)
#   --test_top_dir         (was: test_top_dir, derived from db_top_dir)
#   --checkpoint_top_dir   (was: model_top_dir -- renamed, see note below)
#   --resource_top_dir     (was: spm_top_dir -- renamed, see note below)
#
# Renames:
#   - `model_top_dir` -> `checkpoint_top_dir`: this directory only ever
#     holds `--run_name` subdirectories full of *training checkpoints*
#     (it's passed straight into `TrainingArguments.output_dir`), not
#     model definitions/configs, so "checkpoint" describes its contents
#     more accurately than "model".
#   - `spm_top_dir` -> `resource_top_dir`: the actual path
#     ("<repo_root>/resources/spm") was already a general resource
#     directory, not SPM-specific, and the SPM model filename is still
#     built the same way (`librispeech_unigram_{vocab_size}.model`)
#     underneath it. Renaming acknowledges that other shared resources
#     (LMs, lexicons, etc.) could reasonably live in the same directory
#     later.
_DEFAULT_DB_TOP_DIR = "/mnt/data/database"
_DEFAULT_CHECKPOINT_TOP_DIR = "/mnt/data/home/chanwcom/models"
# Repo-relative, not tied to any one user's home directory: this script
# lives at <repo_root>/scripts/asr/wav2vec2/, and the SPM resources are
# checked into <repo_root>/resources/spm/.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEFAULT_RESOURCE_TOP_DIR = os.path.join(_REPO_ROOT, "resources", "spm")

# GPU hyperparameter presets: hardware/batch-shaped settings only (how big a
# batch and how fast to step), independent of how much data or how long the
# run trains for -- see _FINETUNE_PROFILES below for that axis.
# "4090" is what used to be under `if 1:`, "a100" is what used to be under
# `if 0:`. `--gpu_profile` picks one of these as a starting point; any
# individually-passed CLI flag overrides just that key.
_GPU_PROFILES: Dict[str, Dict[str, Any]] = {
    "4090": dict(
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        learning_rate=1e-4,
        gradient_accumulation_steps=2,
        load_best_model_at_end=False,
    ),
    "a100": dict(
        per_device_train_batch_size=96,
        per_device_eval_batch_size=96,
        learning_rate=4e-4,
        gradient_accumulation_steps=2,
        load_best_model_at_end=False,
    ),
}

# Fine-tuning dataset/schedule presets: how much labeled data to fine-tune
# on, and the step schedule appropriate for that amount.
#   - `train_subdir`: default training data directory, relative to
#     --db_top_dir (used only when --train_top_dir isn't passed explicitly).
#   - `warmup_steps` / `max_steps` / `eval_steps`: schedule sized for that
#     dataset. `save_steps` is auto-synced to `max_steps` below.
# `--finetune_profile` picks one of these; any individually-passed CLI flag
# (--warmup_steps/--max_steps/--eval_steps/--save_steps/--train_top_dir)
# overrides just that value.
_FINETUNE_PROFILES: Dict[str, Dict[str, Any]] = {
    "libri_light_1hr": dict(
        train_subdir="libri_light/1h",
        warmup_steps=1000,
        max_steps=2000,
        eval_steps=500,
    ),
    "libri_light_10hr": dict(
        train_subdir="libri_light/10h",
        warmup_steps=1000,
        max_steps=4000,
        eval_steps=500,
    ),
    "libri_speech_clean_100hr": dict(
        train_subdir="libri_light/train-clean-100",
        warmup_steps=1000,
        max_steps=8000,
        eval_steps=500,
    ),
    # Full LibriSpeech (train-clean-100 + train-clean-360 + train-other-500,
    # ~960h combined). Unlike the profiles above, the shards for this one
    # aren't directly under `train_subdir` -- they're split across three
    # subdirectories, one per split (see `train_shard_subdirs` below, and
    # `sample_util.make_dataset`'s `sub_shard_dirs` param that consumes it).
    "libri_speech_full_960hr": dict(
        train_subdir="libri_speech_webdataset_new_oct_2025",
        train_shard_subdirs=(
            "train-clean-100", "train-clean-360", "train-other-500"),
        warmup_steps=500,
        max_steps=50000,
        eval_steps=500,
    ),
}

# Automatically synchronize save_steps with max_steps for each profile.
for profile in _FINETUNE_PROFILES.values():
    profile["save_steps"] = profile["max_steps"]

class Wav2Vec2SPMTokenizer(PreTrainedTokenizer):
    """Custom Tokenizer for Wav2Vec2 using SentencePiece.

    Inherits from PreTrainedTokenizer to avoid the mandatory vocab.json
    requirement of Wav2Vec2CTCTokenizer.
    """

    def __init__(self, spm_model_path: str, **kwargs: Any):
        """Initializes the tokenizer and loads the SentencePiece model."""
        import sentencepiece as spm
        self.spm_model_path = spm_model_path
        self.sp = spm.SentencePieceProcessor(model_file=spm_model_path)

        # Standard CTC special tokens are passed to the base class.
        super().__init__(
            pad_token="<pad>",
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            **kwargs
        )

    @property
    def vocab_size(self) -> int:
        """Returns the size of the SentencePiece vocabulary."""
        return self.sp.get_piece_size()

    def get_vocab(self) -> Dict[str, int]:
        """Returns the vocabulary as a dictionary for compatibility."""
        return {
            self.sp.id_to_piece(i): i for i in range(self.vocab_size)
        }

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizes text using the SentencePiece engine."""
        return self.sp.encode_as_pieces(text)

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a subword piece to its integer ID."""
        return self.sp.piece_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an integer ID to its subword piece."""
        return self.sp.id_to_piece(index)

    def _decode(self,
                token_ids: List[int],
                group_tokens: bool = True,
                **kwargs: Any) -> str:
        """Decodes IDs with CTC collapse and SentencePiece."""
        if group_tokens:
            token_ids = [k for k, _ in itertools.groupby(token_ids)]

        # Remove padding and ignore index (-100).
        filtered_ids = [
            int(i) for i in token_ids
            if i != self.pad_token_id and i != -100
        ]
        return self.sp.decode(filtered_ids) if filtered_ids else ""

    def save_vocabulary(self,
                        save_directory: str,
                        filename_prefix: Optional[str] = None) -> tuple:
        """Saves the SPM model file. Fixes the NotImplementedError."""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        file_name = "tokenizer.model"
        if filename_prefix:
            file_name = f"{filename_prefix}-{file_name}"

        vocab_file = os.path.join(save_directory, file_name)

        # Copy the original .model file to the checkpoint directory.
        if os.path.abspath(self.spm_model_path) != os.path.abspath(vocab_file):
            shutil.copyfile(self.spm_model_path, vocab_file)

        return (vocab_file,)


def clean_special_tokens(text: str) -> str:
    """Removes start/end-of-sentence markers and extra whitespace."""
    text = re.sub(r'^<s>\s*', '', text)
    text = re.sub(r'\s*</s>$', '', text)
    return text.strip()


def make_compute_metrics(
    processor: AutoProcessor,
) -> Callable[[Any], Dict[str, float]]:
    """Builds a `compute_metrics` closure bound to a specific processor.

    HF `Trainer` calls `compute_metrics(pred)` with a single positional
    argument, so anything else it needs (here, `processor`, to decode
    predicted/label ids back to text) has to come from a closure rather
    than an extra parameter. This factory makes that dependency explicit
    at the call site (`compute_metrics=make_compute_metrics(processor)`)
    instead of relying on a module-level global.
    """

    def compute_metrics(pred) -> Dict[str, float]:
        """Compute Word Error Rate (WER) by mapping sub-labels back to vocab.

        boundary_id and padding are ignored.

        Args:
            pred: A prediction object containing:
                - predictions: Logits of shape (batch, seq, vocab * factor).
                - label_ids: Ground truth IDs (batch, seq).

        Returns:
            A dictionary containing the 'wer' score.
        """
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        # Prepare labels: handle the -100 ignore index.
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions (CTC grouping enabled).
        pred_str = processor.batch_decode(
            pred_ids,
            group_tokens=True,
            skip_special_tokens=False,
        )

        # Decode ground truth labels.
        label_str = processor.batch_decode(
            label_ids,
            group_tokens=False,
            skip_special_tokens=False
        )

        # Clean and calculate WER.
        pred_str = [clean_special_tokens(s) for s in pred_str]
        label_str = [clean_special_tokens(s) for s in label_str]

        wer_metric = evaluate.load("wer")
        wer_score = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer_score}

    return compute_metrics


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
    """Custom Trainer to override loss computation with custom Shc loss."""
    def __init__(self, vocab_size=None, alpha=0.0, beta=0.0,
                peak_preserving=False, gamma=0.0, peak_capping=False,
                smoothing_space="label", dynamic_batching=False,
                *args, **kwargs):
        super().__init__(*args, **kwargs)
        # To include the boundary token at the end.
        self.custom_vocab_size = vocab_size
        self.alpha = alpha
        self.beta = beta
        self.peak_preserving = peak_preserving
        self.gamma = gamma
        self.peak_capping = peak_capping
        self.smoothing_space = smoothing_space
        self.dynamic_batching = dynamic_batching

    def get_train_dataloader(self) -> DataLoader:
        """Builds the training DataLoader.

        Default behavior (`dynamic_batching=False`) is unchanged: fixed
        `per_device_train_batch_size`, with `train_dataset` already
        length-bucketed by `sample_util._length_bucketed_stream` so that
        each `batch_size`-sized chunk DataLoader groups is
        length-homogeneous.

        When `dynamic_batching=True`, `train_dataset` is instead a
        WebDataset pipeline built with `sample_util.DynamicBatchConfig`
        (see `sample_util._dynamic_length_batched_stream`): each item it
        yields is ALREADY one fully collated training batch, with sample
        count varying batch-to-batch so that `sample_count *
        max_input_length` never exceeds a fixed budget -- this is what
        actually bounds peak memory against outlier-length audio, unlike a
        fixed `per_device_train_batch_size` which can still combine
        several long-audio outliers into one OOM-ing batch.

        The standard Trainer path can't express a varying batch size (it
        always asks DataLoader to group a fixed `batch_size` via
        `collate_fn`), so this uses `DataLoader(batch_size=None)` instead,
        which disables DataLoader's own batching/collation and passes each
        already-batched item straight through.
        """
        if not self.dynamic_batching:
            return super().get_train_dataloader()

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
        )
        return self.accelerator.prepare(dataloader)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        with torch.device(inputs["input_values"].device.type):
            target = inputs.pop("labels")
            outputs = model(**inputs)

            input_lengths = inputs["attention_mask"].sum(-1)

            actual_model = model.module if hasattr(model, "module") else model

            logits_lengths = actual_model._get_feat_extract_output_lengths(
                input_lengths)

            logits = outputs["logits"]
            target_lengths = torch.sum(
                (target >= 0).type(torch.int32), axis=1)

            # Custom SHC Loss implementation
            loss = shc_loss.ShcLoss.apply(
                target,
                target_lengths,
                logits.log_softmax(2),
                logits_lengths,
                self.custom_vocab_size,
                self.alpha,
                self.beta,
                self.peak_preserving,
                self.gamma,
                self.peak_capping,
                self.smoothing_space,
            ).mean()

        if return_outputs:
            return loss, outputs
        else:
            return loss


def _fmt_float(x: float) -> str:
    """Formats a float for use in a directory name, e.g. 0.02 -> '0p02'."""
    return str(x).replace(".", "p").replace("-", "neg")


def _batching_suffix(args: argparse.Namespace) -> str:
    """Suffix distinguishing batching-strategy runs in the auto-generated
    run name.

    Without this, e.g. a --dynamic_batching run and a
    --length_bucket_window_mult comparison run with otherwise identical
    --alpha/--beta/--vocab_size/--finetune_profile/--max_steps produce the
    EXACT SAME run name -- meaning the second run silently overwrites the
    first run's checkpoint directory (this actually happened: a
    --length_bucket_window_mult 0 comparison run clobbered a same-named
    fixed-bucketing run's checkpoint-2500). Batching mode isn't a
    hyperparameter of the model, but it's exactly the kind of "otherwise
    identical run" axis people compare against each other, so it needs to
    be part of the name too.
    """
    if args.dynamic_batching:
        suffix = f"_dynbatch{args.max_batch_audio_len}"
        if args.max_dynamic_batch_size is not None:
            suffix += f"_cap{args.max_dynamic_batch_size}"
        return suffix
    return f"_bucket{args.length_bucket_window_mult}"


def _default_run_name(args: argparse.Namespace) -> str:
    """Builds a run name from the *actual* CLI args (used when --run_name
    is not given).

    Unlike the old hardcoded `out_name` string, this reflects whatever
    --alpha/--beta/--max_steps/--vocab_size were actually passed, so two
    runs with different hyperparameters never collide on directory name
    unless they truly are identical runs. --finetune_profile is included
    too, so e.g. 10hr and 100hr runs with the same alpha/beta don't collide.
    The batching strategy (--dynamic_batching / --length_bucket_window_mult)
    is included via `_batching_suffix` for the same reason -- see its
    docstring for the concrete collision this fixes. --seed is included
    too, for the same reason again: multi-seed comparison runs (e.g. 3
    repeats to average out run-to-run training noise) are otherwise
    identical in every other naming input and would overwrite each other.
    """
    suffix = _batching_suffix(args) + f"_seed{args.seed}"
    # Only tagged when non-default, so every existing run name (all of
    # which were label-space) keeps resolving to the same directory.
    if args.smoothing_space != "label":
        suffix += f"_{args.smoothing_space}space"
    if args.vocab_size is not None:
        if args.peak_preserving:
            return (f"{args.finetune_profile}_shc_{args.max_steps}steps_"
                    f"peakpreserving_gamma_{_fmt_float(args.gamma)}"
                    f"_unigram_{args.vocab_size}{suffix}")
        if args.peak_capping:
            return (f"{args.finetune_profile}_shc_{args.max_steps}steps_"
                    f"peakcapping_alpha_{_fmt_float(args.alpha)}"
                    f"_unigram_{args.vocab_size}{suffix}")
        return (f"{args.finetune_profile}_shc_{args.max_steps}steps_alpha_"
                f"{_fmt_float(args.alpha)}_beta_{_fmt_float(args.beta)}"
                f"_unigram_{args.vocab_size}{suffix}")
    return (f"{args.finetune_profile}_ctc_{args.max_steps}steps_default_vocab"
            f"{suffix}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Wav2Vec2 Training with Dynamic Vocab Size")

    parser.add_argument("--vocab_size", type=int, default=None,
                        help="Vocabulary size (e.g., 32, 128). Omit to use "
                             "the default wav2vec2 tokenizer/vocab.")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="(e.g., NZ smoothing coeff.).")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="(e.g., NZ smoothing coeff.).")
    parser.add_argument(
        "--peak_preserving", action="store_true", default=False,
        help="Use apply_peak_preserving_selective_estimated_target_"
             "smoothing (driven by --gamma) instead of the alpha/beta-"
             "driven SETS post-processing. --alpha/--beta are ignored "
             "when this is set.")
    parser.add_argument(
        "--gamma", type=float, default=0.0,
        help="Fraction of each non-peak class's probability mass "
             "redistributed uniformly over the other active, non-peak "
             "classes. Only used when --peak_preserving is set.")
    parser.add_argument(
        "--peak_capping", action="store_true", default=False,
        help="Use apply_peak_capping_selective_estimated_target_"
             "smoothing (driven by --alpha as the confidence-cap "
             "parameter, cap = 1 - alpha) instead of the alpha/beta-"
             "driven SETS post-processing. --beta/--gamma are ignored "
             "when this is set. Ignored if --peak_preserving is set.")
    parser.add_argument(
        "--smoothing_space", type=str, default="label",
        choices=["label", "class"],
        help="Which axis the smoothing is applied to. 'label' (default) "
             "smooths the alignment posterior over blank-augmented label "
             "POSITIONS before scattering to classes -- the historical "
             "behavior every SETS/PP-SETS/PC-SETS result so far used. "
             "'class' scatters first and smooths over actual output "
             "CLASSES. These are different algorithms, not two spellings "
             "of one: in 'label' space a uniform mixin lands as roughly "
             "50%% blank plus an occurrence-weighted prior over only the "
             "classes present in the transcript, whereas in 'class' space "
             "it is genuinely uniform over the vocabulary. See the module "
             "docstring of cwk/loss/pytorch/shc_loss_util.py.")

    # --- Run naming / output location -------------------------------------
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="Directory name for this run's checkpoints, under "
             "--checkpoint_top_dir. If omitted, a name is auto-generated "
             "from --alpha/--beta/--max_steps/--vocab_size (see "
             "_default_run_name()) and printed at startup.")
    parser.add_argument(
        "--checkpoint_top_dir", type=str, default=_DEFAULT_CHECKPOINT_TOP_DIR,
        help="Parent directory under which --run_name is created; this is "
             "where TrainingArguments.output_dir points, i.e. where "
             "training checkpoints actually get written. (Previously "
             "called `model_top_dir`.)")

    # --- Data directories ---------------------------------------------------
    parser.add_argument(
        "--db_top_dir", type=str, default=_DEFAULT_DB_TOP_DIR,
        help="Top-level database directory. Used to derive the default "
             "--train_top_dir / --test_top_dir if those aren't given "
             "explicitly.")
    parser.add_argument(
        "--train_top_dir", type=str, default=None,
        help="Training dataset directory. Defaults to "
             "'<db_top_dir>/<train_subdir of --finetune_profile>'.")
    parser.add_argument(
        "--test_top_dir", type=str, default=None,
        help="Evaluation dataset directory. Defaults to "
             "'<db_top_dir>/libri_speech_webdataset_new_oct_2025/"
             "test-clean'.")
    parser.add_argument(
        "--resource_top_dir", type=str, default=_DEFAULT_RESOURCE_TOP_DIR,
        help="Directory containing shared resources referenced by name, "
             "such as the SentencePiece models "
             "('librispeech_unigram_{vocab_size}.model'). (Previously "
             "called `spm_top_dir`.)")

    # --- Fine-tuning dataset / schedule profile ------------------------------
    parser.add_argument(
        "--finetune_profile", choices=sorted(_FINETUNE_PROFILES.keys()),
        default="libri_light_1hr",
        help="Fine-tuning dataset preset: picks the default --train_top_dir "
             "(under --db_top_dir) plus a warmup_steps/max_steps/eval_steps/"
             "save_steps schedule sized for that amount of data. Any of "
             "--warmup_steps/--max_steps/--eval_steps/--save_steps/"
             "--train_top_dir passed explicitly overrides just that value.")

    # --- GPU / training hyperparameter profile ------------------------------
    parser.add_argument(
        "--gpu_profile", choices=sorted(_GPU_PROFILES.keys()), default="4090",
        help="Preset hyperparameter bundle to start from (matches what "
             "used to be the if-0/if-1 blocks). Any of the flags below, "
             "if passed explicitly, overrides just that value on top of "
             "the chosen profile.")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed. Controls both TrainingArguments' own RNGs "
             "(weight init, etc. -- HF's own default is 42) AND the "
             "training WebDataset stream's length-bucketing/dynamic-"
             "batching per-window shuffle (see sample_util.make_dataset's "
             "`seed` / DynamicBatchConfig.seed), so a single --seed value "
             "gives you a fully independent run for multi-seed comparisons "
             "(e.g. averaging N runs with different --seed to distinguish "
             "a real effect from run-to-run training noise).")
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--load_best_model_at_end", type=bool, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        default=True)
    parser.add_argument("--no_gradient_checkpointing",
                        dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.add_argument("--eval_accumulation_steps", type=int, default=1)

    # --- Data loading parallelism --------------------------------------------
    # With 0 (the default), the WebDataset pipeline -- including the
    # length-bucketing/dynamic-batching window (see below), which has to
    # buffer+decode a full window's worth of audio before it can yield
    # anything -- runs entirely in the main process, serially before each
    # GPU step (i.e. the GPU sits idle while that happens). Setting this >0
    # lets PyTorch DataLoader workers decode/bucket/collate upcoming
    # batches in the background while the GPU is busy with the current one.
    # `wds.WebDataset` already shards its inputs across workers correctly
    # by default (`workersplitter=wds.split_by_worker`, verified against
    # the installed webdataset version) -- no data duplication risk, so
    # this is safe to raise on a machine with CPU cores to spare.
    parser.add_argument(
        "--dataloader_num_workers", type=int, default=0,
        help="Number of DataLoader worker processes for prefetching. 0 "
             "(default) = no overlap between data loading and GPU compute; "
             "the length-bucketing/dynamic-batching window fill (and its "
             "audio decode cost) happens serially before each step. >0 "
             "overlaps that with GPU compute in background worker "
             "processes -- try 4-8 if CPU cores are available "
             "(`nproc`/`uptime` to check headroom).")
    parser.add_argument(
        "--dataloader_persistent_workers", action="store_true", default=False,
        help="Keep worker processes alive between epochs instead of "
             "respawning them (saves worker startup cost on each restart "
             "of the streaming dataset). Only valid with "
             "--dataloader_num_workers > 0.")

    # --- Length-based batch bucketing (training data only) ------------------
    parser.add_argument(
        "--length_bucket_window_mult", type=int, default=50,
        help="Local length-bucketing window size for the *training* "
             "WebDataset stream, as a multiple of "
             "--per_device_train_batch_size (see "
             "sample_util._length_bucketed_stream). Reduces per-batch "
             "padding waste (and thus wall-clock time) by reordering the "
             "sample stream so each batch is drawn from a locally length-"
             "sorted window instead of raw shard order, while still "
             "shuffling the order batches come out in so training doesn't "
             "sweep short-to-long. Pass <= 1 to disable.")

    # --- Dynamic (length-budget) batching (training data only) --------------
    parser.add_argument(
        "--dynamic_batching", action="store_true", default=False,
        help="Replace fixed --per_device_train_batch_size bucketing with "
             "length-budget batching (see sample_util.DynamicBatchConfig): "
             "sample count per training batch varies so that "
             "sample_count * max_input_length_in_batch stays under "
             "--max_batch_audio_len, instead of staying at a fixed count. "
             "This is what actually bounds peak memory against "
             "outlier-length audio -- a fixed batch_size can still OOM "
             "whenever several long-audio outliers land in the same "
             "batch. Mutually exclusive with fixed-size bucketing; "
             "--per_device_train_batch_size is ignored for the training "
             "dataloader when this is set (eval is unaffected).")
    parser.add_argument(
        "--max_batch_audio_len", type=int, default=None,
        help="Required when --dynamic_batching is set. Budget for "
             "sample_count * max_input_values_len within one training "
             "batch, in raw waveform samples (same units as "
             "len(sample['input_values'])). Must be tuned per-GPU/model; "
             "there's no way to derive it analytically. As a starting "
             "point, try (typical --per_device_train_batch_size) * "
             "(a representative max input length for your data).")
    parser.add_argument(
        "--max_dynamic_batch_size", type=int, default=None,
        help="Optional hard cap on sample count per training batch under "
             "--dynamic_batching, even if --max_batch_audio_len would "
             "allow more. Omit for no cap.")
    parser.add_argument(
        "--max_sample_audio_len", type=int, default=None,
        help="Drops any training/eval sample whose raw audio is longer "
             "than this many waveform samples -- a hard safety net "
             "against pathological outliers (e.g. corrupted or "
             "mis-segmented audio) that no amount of batching can absorb "
             "on their own. Applies regardless of --dynamic_batching. "
             "Omit to disable.")

    args = parser.parse_args()

    # Fill in any hyperparameter left as None (i.e. not explicitly passed)
    # from the chosen --gpu_profile and --finetune_profile. The two cover
    # disjoint keys (hardware/batch settings vs. dataset/schedule settings),
    # so the order between them doesn't matter; explicit CLI flags always
    # win over both.
    finetune_profile = _FINETUNE_PROFILES[args.finetune_profile]
    for profile in (_GPU_PROFILES[args.gpu_profile], finetune_profile):
        for key, value in profile.items():
            if key in ("train_subdir", "train_shard_subdirs"):
                continue
            if getattr(args, key) is None:
                setattr(args, key, value)
    args.train_subdir = finetune_profile["train_subdir"]
    # Only set for profiles whose shards are split across multiple
    # subdirectories (e.g. "libri_speech_full_960hr"); None for the rest,
    # meaning sample_util.make_dataset() reads 'shard-*.tar' directly under
    # train_top_dir as before.
    args.train_shard_subdirs = finetune_profile.get("train_shard_subdirs")

    if args.dynamic_batching and args.max_batch_audio_len is None:
        parser.error(
            "--dynamic_batching requires --max_batch_audio_len (there's no "
            "sane default -- it depends on your GPU memory and model).")

    if args.run_name is None:
        args.run_name = _default_run_name(args)
        print(f"[info] --run_name not given; using auto-generated name: "
              f"{args.run_name}")

    return args


def main():
    args = parse_args()

    # Resolve data directories: an explicit --train_top_dir/--test_top_dir
    # always wins; otherwise derive from --db_top_dir, using the
    # --finetune_profile's default train_subdir.
    train_top_dir = args.train_top_dir or os.path.join(
        args.db_top_dir, args.train_subdir)
    test_top_dir = args.test_top_dir or os.path.join(
        args.db_top_dir, "libri_speech_webdataset_new_oct_2025/test-clean")

    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

    # Dynamic configuration based on vocab_size.
    if args.vocab_size is not None:
        spm_name = f"librispeech_unigram_{args.vocab_size}.model"
        spm_model_path = os.path.join(args.resource_top_dir, spm_name)
        # Inject the SPM wrapper into the existing processor structure.
        processor.tokenizer = Wav2Vec2SPMTokenizer(spm_model_path)
    else:
        spm_model_path = None

    # Initialize data collator. Built before the datasets below because
    # --dynamic_batching needs it as the collate_fn baked into the training
    # WebDataset pipeline itself (see DynamicBatchConfig).
    data_collator = DataCollatorCTCWithPadding(
        processor=processor, padding="longest")

    # Dataset preparation. Batching strategy for the *training* stream only
    # -- eval isn't performance/memory-sensitive the same way, so it's
    # always left in raw shard order, consumed with the normal fixed
    # per_device_eval_batch_size DataLoader path.
    if args.dynamic_batching:
        # Length-budget batching: sample count per training batch varies so
        # that sample_count * max_input_length_in_batch stays under
        # --max_batch_audio_len, instead of staying at a fixed count. This
        # is what actually bounds peak memory against outlier-length audio
        # (see MyCtcTrainer.get_train_dataloader). The dataset already
        # yields fully collated batches, so --per_device_train_batch_size
        # is not used for training here.
        train_dataset = sample_util.make_dataset(
            train_top_dir, True, spm_model_path,
            dynamic_batch=sample_util.DynamicBatchConfig(
                collate_fn=data_collator,
                max_batch_length=args.max_batch_audio_len,
                max_batch_size=args.max_dynamic_batch_size,
                window_mult=args.length_bucket_window_mult,
                seed=args.seed),
            sub_shard_dirs=args.train_shard_subdirs,
            max_sample_length=args.max_sample_audio_len)
    else:
        # Fixed-size length bucketing (see --length_bucket_window_mult):
        # trades step-to-step wall-clock variance for lower average padding
        # waste, without changing the batch_size itself.
        train_dataset = sample_util.make_dataset(
            train_top_dir, True, spm_model_path,
            batch_size=args.per_device_train_batch_size,
            length_bucket_window_mult=args.length_bucket_window_mult,
            sub_shard_dirs=args.train_shard_subdirs,
            max_sample_length=args.max_sample_audio_len,
            seed=args.seed)
    test_dataset = sample_util.make_dataset(
        test_top_dir, True, spm_model_path,
        max_sample_length=args.max_sample_audio_len)

    actual_vocab_size = len(processor.tokenizer)

    # Load model with dynamic vocab size.
    model = AutoModelForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=actual_vocab_size,
        ignore_mismatched_sizes=True,
    )

    output_dir = os.path.join(args.checkpoint_top_dir, args.run_name)
    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=args.seed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        eval_strategy="steps",
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_accumulation_steps=args.eval_accumulation_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_persistent_workers=(
            args.dataloader_persistent_workers
            if args.dataloader_num_workers > 0 else False),
    )

    # Initialize trainer and start training.
    trainer = MyCtcTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=processor,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(processor),
        vocab_size=args.vocab_size,
        alpha=args.alpha,
        beta=args.beta,
        peak_preserving=args.peak_preserving,
        gamma=args.gamma,
        peak_capping=args.peak_capping,
        smoothing_space=args.smoothing_space,
        dynamic_batching=args.dynamic_batching
    )

    trainer.train()


if __name__ == "__main__":
    main()
