# pylint: disable=import-error, no-member
"""Inference / WER-evaluation script matching the SHC/CTC training script.

This replaces the old TensorFlow-TFRecord-based inference script. It was
out of sync with the current training setup in a few concrete ways that
are fixed here:

  1. TOKENIZER MISMATCH (the important bug):
     Training now uses a custom SentencePiece tokenizer
     (`Wav2Vec2SPMTokenizer`) whenever `--vocab_size` is given, but the old
     inference script always loaded the *default* wav2vec2 tokenizer via
     `transformers.AutoTokenizer.from_pretrained("facebook/wav2vec2-base")`.
     For any checkpoint trained with a non-default `--vocab_size`, that
     tokenizer's vocabulary doesn't match the model's output classes at
     all, so decoding would be garbage. This script defines/uses the same
     `Wav2Vec2SPMTokenizer` class as training and builds it the same way
     (`--vocab_size` -> spm model path -> tokenizer), so evaluation always
     uses the exact tokenizer the checkpoint was trained with.

  2. DATA PIPELINE:
     Swapped the standalone TensorFlow/TFRecord pipeline
     (`speech_data_helper.SpeechDataToWave` + `tf.data.TFRecordDataset`)
     for `sample_util.make_dataset(...)` + `DataCollatorCTCWithPadding`,
     i.e. the exact same dataset/collator objects used during training, so
     evaluation numbers are directly comparable to the training-time
     `compute_metrics` WER.

  3. CHECKPOINT PATH:
     No longer hardcoded to an old experiment directory. Passed explicitly
     via `--checkpoint_dir`.

  4. DECODER:
     Both decoding strategies from the old script are kept, selectable via
     `--decoder {pipeline, beam_search}`:
       - "pipeline": HuggingFace `transformers.pipeline(...)` (greedy CTC).
       - "beam_search": torchaudio's `ctc_decoder`.
     The old script's third, TensorFlow-based `tf.nn.ctc_beam_search_decoder`
     branch (with its manual blank-index-swap hack) was dropped -- it was
     already commented out/unused in the original, redundant with the
     torchaudio beam-search path, and specific to the *default* wav2vec2
     vocab layout (assumed blank at a fixed index), which doesn't
     generalize to the SPM tokenizer's vocab layout.

     Two correctness fixes were made to the beam-search path while porting
     it over:
       - It used to build hypothesis text as
         `"".join(tokens[i] for i in output.tokens).replace("|", " ")`,
         which only works for the original wav2vec2 vocab (where "|" is
         the literal word-boundary token). That doesn't hold for the SPM
         tokenizer (which uses a "_" (U+2581) word-boundary marker
         instead). Replaced with `processor.tokenizer.decode(...)`, which
         is correct for either tokenizer.
       - It fed raw `logits` into the decoder. torchaudio's `ctc_decoder`
         expects emissions in *log-probability* form, so `log_softmax` is
         now applied first.

  TODO / please verify against your actual `sample_util.make_dataset`
  implementation:
    - This script assumes each dataset example is a dict with at least
      "input_values" (raw waveform, matching what
      `DataCollatorCTCWithPadding`/`processor.pad(...)` expects) and
      "labels" (already-tokenized label ids), i.e. the same shape of
      example that `MyCtcTrainer`/the HF `Trainer` consumed during
      training. If `sample_util.make_dataset` returns a different key
      layout for the *test* split specifically, adjust
      `DataCollatorCTCWithPadding.__call__` or add a small adapter.
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
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# Third-party imports
import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchaudio.models import decoder as torchaudio_decoder
from transformers import (AutoModelForCTC, AutoProcessor,
                          PreTrainedTokenizer, pipeline)

# Custom imports
from common import sample_util

# Same directory conventions as the training script.
db_top_dir = "/mnt/data/database"
test_top_dir = os.path.join(
    db_top_dir, "libri_speech_webdataset_new_oct_2025/test-clean")
spm_top_dir = ("/mnt/data/home/chanwcom/local_repository/"
              "cognitive_workflow_kit_emnlp_2026/run/resources")


# -----------------------------------------------------------------------
# Kept identical to the training script (train_shc.py) so that training and
# inference can never silently drift apart. If this class changes in
# training, mirror the change here too -- or better, factor it out into a
# shared module both scripts import from.
# -----------------------------------------------------------------------
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

        if os.path.abspath(self.spm_model_path) != os.path.abspath(vocab_file):
            shutil.copyfile(self.spm_model_path, vocab_file)

        return (vocab_file,)


# -----------------------------------------------------------------------
# Kept identical to the training script.
# -----------------------------------------------------------------------
@dataclass
class DataCollatorCTCWithPadding:
    """Data collator that dynamically pads inputs and labels for CTC training."""

    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]}
                          for feature in features]
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
            return_attention_mask=True
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch


def clean_special_tokens(text: str) -> str:
    """Removes start/end-of-sentence markers and extra whitespace.

    Same helper as `compute_metrics()` in the training script, kept in
    sync so hypotheses/references are normalized identically.
    """
    text = re.sub(r'^<s>\s*', '', text)
    text = re.sub(r'\s*</s>$', '', text)
    return text.strip()


def build_processor(vocab_size: Optional[int]) -> AutoProcessor:
    """Builds the same processor/tokenizer combination used in training.

    Args:
        vocab_size: The `--vocab_size` value the checkpoint was trained
            with. Must match; there's no way to auto-detect this from the
            checkpoint alone since `Wav2Vec2SPMTokenizer` isn't registered
            with the `AutoTokenizer` machinery. Pass None to fall back to
            the default wav2vec2 tokenizer (i.e. a checkpoint trained
            without `--vocab_size`).

    Returns:
        (processor, spm_model_path). `spm_model_path` is None when
        `vocab_size` is None.
    """
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
    if vocab_size is None:
        return processor, None

    spm_name = f"librispeech_unigram_{vocab_size}.model"
    spm_model_path = os.path.join(spm_top_dir, spm_name)
    processor.tokenizer = Wav2Vec2SPMTokenizer(spm_model_path)
    return processor, spm_model_path


def build_beam_search_decoder(processor, beam_size: int = 50):
    """Builds a torchaudio CTC beam search decoder for the given tokenizer.

    Returns:
        (beam_decoder, synthetic_sil_token_id). `synthetic_sil_token_id` is
        None when the real "|" word-boundary token is available (the
        default wav2vec2 vocab case -- its tokenizer's own `decode()`
        already knows how to turn "|" into a space, so nothing needs to be
        stripped). It's the id of our unk-token stand-in otherwise (the
        SentencePiece vocab case), so the caller knows which id to filter
        out of the hypothesis before decoding to text.
    """
    vocab = processor.tokenizer.get_vocab()
    tokens = sorted(vocab, key=lambda token: vocab[token])

    # torchaudio's ctc_decoder requires `sil_token` to be an entry that
    # literally exists in `tokens` (it does tokens_dict.get_index(sil_token)
    # internally). The default "|" only exists in the *original* wav2vec2
    # vocab, where "|" is the literal word-boundary token. The
    # SentencePiece vocab uses "_" (U+2581) *embedded inside* word-initial
    # pieces (e.g. "_hello") instead of a standalone delimiter token, so
    # "|" isn't present there and ctc_decoder(...) raises
    # `ValueError: Unknown entry in dictionary: '|'`.
    #
    # We're decoding lexicon-free (lexicon=None) with the default
    # sil_score=0, so the exact choice of sil_token has no real effect on
    # scoring here -- it just needs to be *some* token that's guaranteed
    # to exist in the vocab. Fall back to unk_token (always present as a
    # special token) when "|" isn't in the vocab.
    #
    # NOTE: the lexicon-free decoder inserts this sil_token at utterance
    # boundaries. For the real "|" case that's fine and expected (the
    # tokenizer's decode() converts "|" -> " "). For our unk-token
    # stand-in, SentencePiece's decode() has no such special-casing and
    # will render it as the literal unk piece (typically "⁇"), leaking
    # into the output text -- so the caller must strip
    # `synthetic_sil_token_id` from the hypothesis before decoding.
    using_real_sil_token = "|" in vocab
    sil_token = "|" if using_real_sil_token else processor.tokenizer.unk_token
    synthetic_sil_token_id = (
        None if using_real_sil_token
        else processor.tokenizer.convert_tokens_to_ids(sil_token))

    beam_decoder = torchaudio_decoder.ctc_decoder(
        lexicon=None,
        tokens=tokens,
        lm=None,
        nbest=1,
        beam_size=beam_size,
        blank_token=processor.tokenizer.pad_token,
        sil_token=sil_token,
    )
    return beam_decoder, synthetic_sil_token_id


def decode_batch_pipeline(asr_pipeline, input_values, attention_mask):
    """Decodes a padded batch with the HF ASR pipeline.

    The pipeline expects raw (unpadded) waveforms, so each example is
    trimmed back to its real length using `attention_mask` before being
    handed to the pipeline -- otherwise trailing padding would be fed
    through the feature extractor / model as if it were real audio.
    """
    lengths = attention_mask.sum(dim=-1).tolist()
    waveforms = [
        input_values[i, :lengths[i]].cpu().numpy()
        for i in range(input_values.shape[0])
    ]
    outputs = asr_pipeline(waveforms)
    return [clean_special_tokens(o["text"]) for o in outputs]


def decode_batch_beam_search(model, processor, beam_decoder,
                             synthetic_sil_token_id, input_values,
                             attention_mask):
    """Decodes a padded batch with the torchaudio CTC beam search decoder."""
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    # torchaudio's ctc_decoder expects log-probabilities, not raw logits.
    log_probs = logits.log_softmax(dim=-1).to(torch.float32).cpu()

    input_lengths = attention_mask.sum(dim=-1)
    output_lengths = model._get_feat_extract_output_lengths(
        input_lengths).cpu()

    beam_outputs = beam_decoder(log_probs, output_lengths)

    hyp_list = []
    for utterance_hyps in beam_outputs:
        token_ids = utterance_hyps[0].tokens.tolist()
        if synthetic_sil_token_id is not None:
            # Strip the unk-token stand-in the decoder inserted as a
            # boundary marker (see build_beam_search_decoder) -- otherwise
            # it leaks into the text as a literal "⁇".
            token_ids = [t for t in token_ids if t != synthetic_sil_token_id]
        # Reuse the tokenizer's own decode logic (handles both the default
        # wav2vec2 vocab and the SentencePiece vocab correctly, unlike the
        # old "|" -> " " string-replace hack).
        text = processor.tokenizer.decode(token_ids, group_tokens=False)
        hyp_list.append(clean_special_tokens(text))
    return hyp_list


def configure_tensorflow_gpu_memory_growth():
    """Prevents TensorFlow from grabbing the entire GPU up front.

    We're running PyTorch (the model) and TensorFlow (just for
    tf.nn.ctc_beam_search_decoder) side by side on the same GPU. Without
    this, TF's default behavior is to pre-allocate ~all GPU memory for
    itself the moment it touches the device, starving the PyTorch model.
    Same fix as the original inference script used.
    """
    import tensorflow as tf  # local import: only needed for this decoder
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Must be set before GPUs have been initialized; if we're too
            # late, just report it and carry on.
            print(f"[warn] could not set TF GPU memory growth: {e}")


def decode_batch_tf_beam_search(model, processor, input_values,
                                attention_mask, beam_width=50, top_paths=1):
    """Decodes a padded batch with tf.nn.ctc_beam_search_decoder.

    This is the same decoding backend as the (previously disabled, `if 0:`)
    TensorFlow branch in the original inference script, now wired up as a
    proper `--decoder tf_beam_search` option and generalized to work with
    either the default wav2vec2 vocab or the custom SentencePiece vocab.

    Inference-only, framework-mixing note: the model itself still runs
    entirely in PyTorch. We only convert the resulting logits tensor to a
    TF tensor for this one decode call (numpy round-trip), then convert
    the decoded ids straight back to Python ints -- nothing about the
    model, gradients, or training touches TensorFlow at any point, so
    there's no cross-framework autograd concern here.

    tf.nn.ctc_beam_search_decoder hardcodes blank_index = num_classes - 1
    with no way to override this (a long-standing TensorFlow
    inconsistency -- see tensorflow/tensorflow#42993, #40727, #32903).
    Our blank is at `processor.tokenizer.pad_token_id`, which usually is
    *not* the last class index. Rather than reordering the tokenizer's
    whole vocabulary, we just swap those two class *columns* in the
    logits before calling TF, and invert the swap on the decoded ids
    afterward. (This is the same trick the original script applied
    manually and unconditionally; here it's computed from the actual
    tokenizer so it's correct for both vocab layouts, and is a no-op when
    blank already happens to be the last index.)

    Also note: despite the API docstring saying "logits", TF's own
    ctc_beam_search_decoder actually expects softmax probabilities
    already applied (see tensorflow/tensorflow#42151 -- the docs are
    wrong; tf.keras.backend.ctc_decode, which wraps this same op, expects
    softmax output too). We apply torch.softmax (not log_softmax, not raw
    logits) before handing anything to TF.
    """
    import tensorflow as tf  # local import: only needed for this decoder

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits  # (B, T, C)

    num_classes = logits.shape[-1]
    blank_id = processor.tokenizer.pad_token_id
    last_id = num_classes - 1

    if blank_id != last_id:
        logits = logits.clone()
        blank_col = logits[..., blank_id].clone()
        logits[..., blank_id] = logits[..., last_id]
        logits[..., last_id] = blank_col

    # tf.nn.ctc_beam_search_decoder's docstring says it takes "logits", but
    # that's a long-standing documentation bug (see
    # tensorflow/tensorflow#42151): it actually expects softmax
    # probabilities already applied (same as what tf.keras.backend.ctc_decode
    # feeds it). Passing raw logits breaks the internal probability
    # accumulation and produces garbled output -- feed softmax, not raw
    # logits, and NOT log-softmax either.
    probs = torch.softmax(logits, dim=-1)

    # tf.nn.ctc_beam_search_decoder expects time-major [T, B, C].
    probs_time_major = probs.transpose(0, 1).to(torch.float32).cpu().numpy()
    tf_probs = tf.convert_to_tensor(probs_time_major)

    input_lengths = attention_mask.sum(dim=-1)
    output_lengths = model._get_feat_extract_output_lengths(input_lengths)
    tf_lengths = tf.convert_to_tensor(
        output_lengths.cpu().numpy().astype("int32"))

    decoded, _log_probs = tf.nn.ctc_beam_search_decoder(
        tf_probs, tf_lengths, beam_width=beam_width, top_paths=top_paths)

    # decoded[0]: SparseTensor holding the top hypothesis for every example
    # in the batch. Densify with -1 padding so we can strip it back out.
    dense = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()

    hyp_list = []
    for row in dense:
        token_ids = [int(t) for t in row if t != -1]
        if blank_id != last_id:
            # Invert the column swap: anything TF emits as `blank_id` is
            # really the token that originally lived at `last_id` (TF
            # never emits its own blank position -- CTC blanks are always
            # stripped internally -- so no other case can occur here).
            token_ids = [last_id if t == blank_id else t for t in token_ids]
        text = processor.tokenizer.decode(token_ids, group_tokens=False)
        hyp_list.append(clean_special_tokens(text))
    return hyp_list


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wav2Vec2/SHC inference and WER evaluation")
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Path to a trained model checkpoint directory "
             "(e.g. .../model_2000_steps_.../checkpoint-2000).")
    parser.add_argument(
        "--vocab_size", type=int, default=None,
        help="Vocabulary size the checkpoint was trained with (e.g. 32, "
             "128). Must match the --vocab_size used for training this "
             "checkpoint; omit only for checkpoints trained without "
             "--vocab_size (default wav2vec2 vocab).")
    parser.add_argument(
        "--decoder", choices=["pipeline", "beam_search", "tf_beam_search"],
        default="pipeline",
        help="Decoding strategy: HF pipeline() greedy CTC decoding; "
             "torchaudio (flashlight) CTC beam search; or "
             "tf.nn.ctc_beam_search_decoder (requires tensorflow "
             "installed). Note: --beam_size only applies to the two "
             "beam-search options -- the HF pipeline() path always does "
             "greedy (argmax) decoding and has no beam-search option "
             "unless you attach a separate pyctcdecode decoder, which "
             "this script doesn't do.")
    parser.add_argument(
        "--beam_size", type=int, default=50,
        help="Beam width for --decoder beam_search / tf_beam_search. "
             "Ignored for --decoder pipeline.")
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Evaluation batch size.")
    parser.add_argument(
        "--max_examples", type=int, default=None,
        help="Optional cap on the number of evaluation examples, for a "
             "quick sanity check run.")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    processor, spm_model_path = build_processor(args.vocab_size)

    # Same dataset construction call as training's test_dataset.
    test_dataset = sample_util.make_dataset(test_top_dir, True, spm_model_path)
    data_collator = DataCollatorCTCWithPadding(
        processor=processor, padding="longest")
    dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, collate_fn=data_collator)

    model = AutoModelForCTC.from_pretrained(args.checkpoint_dir).to(args.device)
    model.eval()

    asr_pipeline = None
    beam_decoder = None
    synthetic_sil_token_id = None
    if args.decoder == "pipeline":
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
            device=(0 if args.device == "cuda" else -1),
            # Without this, passing a list of waveforms to asr_pipeline(...)
            # still processes them one at a time internally -- batch_size
            # here is what actually makes it group --batch_size waveforms
            # into a single forward pass.
            batch_size=args.batch_size,
        )
    elif args.decoder == "beam_search":
        beam_decoder, synthetic_sil_token_id = build_beam_search_decoder(
            processor, beam_size=args.beam_size)
    else:  # tf_beam_search
        # Fail fast (before loading the whole dataset/eval loop) if
        # tensorflow isn't installed, rather than partway through.
        try:
            import tensorflow  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "--decoder tf_beam_search requires tensorflow to be "
                "installed (`pip install tensorflow`)."
            ) from e
        configure_tensorflow_gpu_memory_growth()

    ref_list: List[str] = []
    hyp_list: List[str] = []
    num_examples = 0
    start_time = time.time()

    for batch in dataloader:
        input_values = batch["input_values"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["labels"]

        # Reference text, using the exact same convention as
        # compute_metrics() in the training script (map -100 -> pad,
        # group_tokens=False since labels are not CTC-collapsed, then
        # strip <s>/</s>).
        label_ids = labels.clone().numpy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        batch_ref = processor.batch_decode(
            label_ids, group_tokens=False, skip_special_tokens=False)
        batch_ref = [clean_special_tokens(s) for s in batch_ref]

        if args.decoder == "pipeline":
            batch_hyp = decode_batch_pipeline(
                asr_pipeline, input_values, attention_mask)
        elif args.decoder == "beam_search":
            batch_hyp = decode_batch_beam_search(
                model, processor, beam_decoder, synthetic_sil_token_id,
                input_values, attention_mask)
        else:  # tf_beam_search
            batch_hyp = decode_batch_tf_beam_search(
                model, processor, input_values, attention_mask,
                beam_width=args.beam_size)

        for ref, hyp in zip(batch_ref, batch_hyp):
            print(f"REF: {ref}")
            print(f"HYP: {hyp}")
            ref_list.append(ref)
            hyp_list.append(hyp)
            num_examples += 1

        if args.max_examples is not None and num_examples >= args.max_examples:
            break

    print(f"Elapsed time during the experiment: {time.time() - start_time:.2f} s")

    wer_metric = evaluate.load("wer")
    result = wer_metric.compute(references=ref_list, predictions=hyp_list)
    print(result)


if __name__ == "__main__":
    main()
