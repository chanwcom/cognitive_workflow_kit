# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard library imports
import glob
import io
import os
import random
from dataclasses import dataclass
from typing import (Any, Callable, Dict, Iterator, List, Optional, Sequence,
                    Union)

# Third-party imports
import numpy as np
import sentencepiece as spm
import torchaudio
import torch
import soundfile as sf
import webdataset as wds
from transformers import AutoProcessor

# Define processor globally
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")


def preprocess_sample(
    sample: Dict[str, Any],
    do_tokenization: bool = True,
    tokenizer_obj: Optional[Union[spm.SentencePieceProcessor, Any]] = None
) -> Dict[str, Any]:
    """Preprocess a single raw sample from the WebDataset.

    This function loads the waveform from the raw bytes using torchaudio,
    extracts features using the processor's feature extractor, and tokenizes
    the transcript text.
 
    Args:
        sample: Dictionary containing 'audio' (bytes) and 'text'.
        do_tokenization: Whether to convert text to token IDs.
        tokenizer_obj: Optional tokenizer (SentencePiece or HF). If None,
            uses the global processor.tokenizer.

    Returns:
        Dict with 'input_values' (audio features) and 'labels' (token IDs).            
            - 'input_values': processed audio feature tensor.
            - 'labels': list of token IDs corresponding to the transcript.
    """
    # Load waveform from raw bytes using torchaudio
    waveform, sample_rate = torchaudio.load(io.BytesIO(sample["audio"]))
    input_values = processor.feature_extractor(
        waveform[0], sampling_rate=sample_rate
    ).input_values[0]

    # Handle text decoding from bytes or string
    text = sample["text"]
    if isinstance(text, bytes):
        text = text.decode("utf-8").strip()
    else:
        text = text.strip()

    # Tokenize text using the provided tokenizer or default processor
    if do_tokenization:
        if tokenizer_obj is None:
            # Hack
            # TODO(chanwcom)
            # Need to make the following changes:
            # 1. Prepends and appends <s> and </s> only when those are required as options.
            # 2. Even if this option is enabled, check whether <s> and </s> are already
            #     there, and in this case, do not pre-pend or append.
            text = f"<s> {text} </s>"
            labels = processor.tokenizer(text).input_ids
        elif isinstance(tokenizer_obj, spm.SentencePieceProcessor):
            # SentencePiece encoding returns a list of integers
            labels = tokenizer_obj.encode(
                text, add_bos=True, add_eos=True, out_type=int)
        else:
            labels = tokenizer_obj(text).input_ids
    else:
        labels = text

    return {"input_values": input_values, "labels": labels}


def _length_bucketed_stream(
    samples: Iterator[Dict[str, Any]],
    batch_size: int,
    window_mult: int = 50,
    seed: int = 0,
) -> Iterator[Dict[str, Any]]:
    """Reorders a stream of samples so that batches are length-homogeneous.

    This is a WebDataset pipeline stage (an iterator-to-iterator function,
    added via `.compose()`): it does NOT batch samples itself. Instead it
    relies on the fact that the downstream `DataLoader(batch_size=...)`
    groups every `batch_size` *consecutive* items from this stream into one
    batch -- so reordering the stream so that consecutive runs of
    `batch_size` samples have similar `input_values` length is enough to
    make the resulting batches length-homogeneous, with zero changes needed
    to the data collator or Trainer.

    batch_size itself is NOT changed by this -- it only reduces the amount
    of wasted padding within each fixed-size batch, i.e. this trades
    "almost every batch pays close to the dataset's worst-case padded
    length" (what plain shuffled/shard-order streaming gives you, since the
    max of `batch_size` random draws tends to land near the length
    distribution's tail once `batch_size` is reasonably large) for "only
    batches that land in the long part of a window pay that cost, batches
    of mostly-short samples finish much faster" -- see the earlier
    discussion this implements: it changes step wall-clock time, not
    per-step memory usage.

    A true global sort isn't possible (or desirable) for a streaming
    dataset, so this only sorts within a local buffer ("window") of
    `batch_size * window_mult` samples at a time. Importantly, the *order
    in which the resulting length-homogeneous batches are yielded* is
    shuffled per window -- without this, the stream would sweep from
    shortest to longest (or vice versa) within every window, which would
    make the data distribution seen by the model non-stationary over the
    course of training (e.g. a run of steps seeing only short utterances,
    then a run seeing only long ones) instead of just reordering for
    padding efficiency.

    Args:
        samples: Iterator of already-preprocessed sample dicts (must
            contain "input_values", whose length is used as the sort key).
        batch_size: Must match the DataLoader's batch_size exactly --
            this is what determines where each "batch" boundary falls once
            the reordered stream is chunked downstream.
        window_mult: Buffer size = batch_size * window_mult samples.
            Larger windows -> more length-homogeneous batches (closer to a
            global sort), at the cost of more samples held in memory at
            once and a coarser-grained/less frequent reshuffle.
        seed: Seed for the per-window batch-order shuffle.

    Yields:
        The same samples, reordered.
    """
    rng = random.Random(seed)
    window_size = max(batch_size * window_mult, batch_size)

    def _flush(buf: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        buf.sort(key=lambda s: len(s["input_values"]))
        batches = [buf[i:i + batch_size] for i in range(0, len(buf), batch_size)]
        rng.shuffle(batches)
        for batch in batches:
            yield from batch

    buf: List[Dict[str, Any]] = []
    for sample in samples:
        buf.append(sample)
        if len(buf) >= window_size:
            yield from _flush(buf)
            buf = []
    if buf:
        yield from _flush(buf)


@dataclass
class DynamicBatchConfig:
    """Configures length-budget ("dynamic") batching (see
    `_dynamic_length_batched_stream`).

    Unlike `batch_size`-based bucketing, which keeps the *sample count* per
    batch fixed and lets padding vary, dynamic batching keeps an upper bound
    on the *padded batch cost* (`sample_count * max_length_in_batch`) and
    lets the sample count vary instead. This is what actually caps peak
    memory: a fixed `batch_size` still OOMs whenever enough long-audio
    outliers land in the same batch, whereas a length budget can never be
    exceeded regardless of which samples happen to land together -- a batch
    of a few long utterances and a batch of many short ones cost about the
    same.

    Attributes:
        collate_fn: Called once per assembled batch (a list of sample
            dicts) to produce the final padded/tensorized batch dict, e.g.
            a `DataCollatorCTCWithPadding` instance. Injected rather than
            imported here so this module stays independent of any
            particular model's collator.
        max_batch_length: Budget for `sample_count * max_input_values_len`
            within one batch, in the same units as `len(sample
            ["input_values"])` (raw waveform samples for wav2vec2's feature
            extractor). Must be tuned per-GPU/model; there's no way to
            derive it analytically here.
        max_batch_size: Optional hard cap on sample count per batch, even if
            the length budget would allow more (e.g. to bound BatchNorm-like
            statistics or op counts). None = uncapped.
        window_mult: Local sort-window size, as a multiple of an assumed
            "typical" batch -- reuses the same window/shuffle mechanism as
            `_length_bucketed_stream`; see its docstring for why a window
            (not a global sort) and why the per-window shuffle matters.
        seed: Seed for the per-window batch-order shuffle.
    """
    collate_fn: Callable[[List[Dict[str, Any]]], Any]
    max_batch_length: int
    max_batch_size: Optional[int] = None
    window_mult: int = 50
    seed: int = 0


def _dynamic_length_batched_stream(
    samples: Iterator[Dict[str, Any]],
    config: DynamicBatchConfig,
) -> Iterator[Any]:
    """Groups a sample stream into variable-size, length-budget-capped
    batches, and yields the already-collated batch (not raw samples).

    This is a WebDataset pipeline stage (added via `.compose()`), and unlike
    `_length_bucketed_stream` it DOES batch (and collate) samples itself --
    each item this yields is one full training batch. This means the
    resulting dataset must be consumed with `DataLoader(batch_size=None)`
    (automatic batching disabled), since sample count varies batch to batch
    and there is no fixed `batch_size` for DataLoader to chunk by.

    Batches are built by first locally sorting a window of samples by
    length (same window/shuffle scheme as `_length_bucketed_stream`, see
    its docstring), then greedily packing the sorted window: walking from
    shortest to longest, a sample is added to the current batch unless
    doing so would push `(current_batch_size + 1) * this_sample_length`
    over `config.max_batch_length` (or `config.max_batch_size`), in which
    case the current batch is closed and a new one started with that
    sample. Because the window is sorted ascending, the newest sample
    added is always the longest one in the batch so far, so its length
    alone (not a running max) is the right multiplier -- no need to track
    a separate running max.

    A single sample whose own length already exceeds `max_batch_length`
    unavoidably becomes a batch of size 1 that still exceeds the nominal
    budget; this only bounds *combining* outliers with others, it is not a
    substitute for filtering genuinely pathological samples (see
    `make_dataset`'s `max_sample_length`).

    Args:
        samples: Iterator of already-preprocessed sample dicts (must
            contain "input_values", whose length is used as the packing
            key).
        config: See `DynamicBatchConfig`.

    Yields:
        `config.collate_fn(batch)` for each assembled batch, in per-window
        shuffled order (not sorted order -- see `_length_bucketed_stream`
        for why).
    """
    rng = random.Random(config.seed)
    # Window size, in raw samples, for the local sort/pack buffer. Sized as
    # a multiple of an ASSUMED typical batch size, the same idea as
    # `_length_bucketed_stream`'s `batch_size * window_mult` -- but unlike
    # that function, there's no real `batch_size` here to multiply by (that
    # is the whole point of dynamic batching), so `config.max_batch_size`
    # is used as a stand-in WHEN the caller set one. When it's None (the
    # common case -- no hard cap), falling back to `1 * window_mult` was a
    # bug: it collapsed the window to just `window_mult` (default 50)
    # samples, nowhere near enough to hold even one budget-sized batch, let
    # alone the many batches this needs to meaningfully sort/shuffle
    # across. That produced far smaller, far more numerous batches than
    # the budget actually allows -- with no correctness impact (the packer
    # still enforces `max_batch_length` no matter how small the window is,
    # so this was never an OOM risk), but real throughput impact (more,
    # smaller optimizer steps to process the same data, i.e. slower
    # wall-clock) and, more importantly, an effective-batch-size /
    # effective-samples-seen mismatch against the (fixed) --max_steps
    # schedule the --finetune_profile presets assume.
    # `_ASSUMED_BATCH_SIZE_FOR_WINDOWING` is a heuristic fallback for that
    # unset case, not a measured value.
    _ASSUMED_BATCH_SIZE_FOR_WINDOWING = 64
    window_size = (
        (config.max_batch_size or _ASSUMED_BATCH_SIZE_FOR_WINDOWING)
        * config.window_mult)

    def _pack(sorted_buf: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        batches: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        for sample in sorted_buf:
            length = len(sample["input_values"])
            prospective_count = len(current) + 1
            over_length_budget = (
                prospective_count * length > config.max_batch_length)
            over_size_cap = (
                config.max_batch_size is not None
                and prospective_count > config.max_batch_size)
            if current and (over_length_budget or over_size_cap):
                batches.append(current)
                current = [sample]
            else:
                current.append(sample)
        if current:
            batches.append(current)
        return batches

    def _flush(buf: List[Dict[str, Any]]) -> Iterator[Any]:
        buf.sort(key=lambda s: len(s["input_values"]))
        batches = _pack(buf)
        rng.shuffle(batches)
        for batch in batches:
            yield config.collate_fn(batch)

    buf: List[Dict[str, Any]] = []
    for sample in samples:
        buf.append(sample)
        if len(buf) >= window_size:
            yield from _flush(buf)
            buf = []
    if buf:
        yield from _flush(buf)


def make_dataset(
    data_dir: str,
    do_tokenization: bool = True,
    spm_model_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    length_bucket_window_mult: int = 50,
    sub_shard_dirs: Optional[Sequence[str]] = None,
    max_sample_length: Optional[int] = None,
    dynamic_batch: Optional[DynamicBatchConfig] = None,
) -> wds.WebDataset:
    """Create a WebDataset pipeline with optional SentencePiece support.

    By default, reads all shards named 'shard-*.tar' directly under
    `data_dir`. If `sub_shard_dirs` is given, instead reads
    'shard-*.tar' from each of `data_dir/<sub_shard_dirs[i]>` and combines
    them into a single shard list -- e.g. for a directory laid out as:
        data_dir/train-clean-100/shard-*.tar
        data_dir/train-clean-360/shard-*.tar
        data_dir/train-other-500/shard-*.tar
    passing `sub_shard_dirs=("train-clean-100", "train-clean-360",
    "train-other-500")` trains on the union of all three splits.

    It extracts 'wav' and 'txt' entries as tuples, converts them into
    dictionaries, and applies the preprocessing function.

    Args:
        data_dir: Path to directory containing 'shard-*.tar' (or, if
            `sub_shard_dirs` is given, containing one subdirectory per
            entry in `sub_shard_dirs`, each with its own 'shard-*.tar').
        do_tokenization: Whether to apply tokenization during mapping.
        spm_model_path: Path to the SentencePiece *.model file.
        batch_size: Fixed-size length bucketing (see
            `_length_bucketed_stream`). Mutually exclusive with
            `dynamic_batch` -- pass at most one of the two. If given (and
            `length_bucket_window_mult > 1`), reorders the sample stream so
            that consecutive `batch_size`-sized chunks are
            length-homogeneous. Must match the batch_size the resulting
            dataset will actually be consumed with (e.g.
            `TrainingArguments.per_device_train_batch_size`) for the
            bucketing to line up with real batch boundaries. Omit (or pass
            0/None) to leave the stream in raw shard order.
        length_bucket_window_mult: Local sort-window size, as a multiple of
            `batch_size`. Ignored if `batch_size` is not given. Pass <= 1
            to disable bucketing even if `batch_size` is given.
        sub_shard_dirs: Optional list of subdirectory names under
            `data_dir`, each containing its own 'shard-*.tar' files, to be
            combined into one shard list. Omit to read 'shard-*.tar'
            directly under `data_dir` (the original single-split layout).
        max_sample_length: If given, drops any sample whose "input_values"
            is longer than this (a hard safety cap on pathological
            outliers, e.g. corrupted/mis-segmented audio). Independent of
            `batch_size`/`dynamic_batch`; applies in either mode.
        dynamic_batch: If given, enables length-budget ("dynamic") batching
            instead of fixed-`batch_size` bucketing (see
            `_dynamic_length_batched_stream` / `DynamicBatchConfig`).
            Mutually exclusive with `batch_size`. The returned dataset then
            yields already-collated BATCHES (not individual samples), so it
            must be consumed with `DataLoader(batch_size=None)` --
            `MyCtcTrainer.get_train_dataloader` does this when
            `dynamic_batching=True`.

    Returns:
        A prepared WebDataset pipeline.
    """
    if batch_size and dynamic_batch is not None:
        raise ValueError(
            "batch_size (fixed-size bucketing) and dynamic_batch "
            "(length-budget batching) are mutually exclusive; pass at "
            "most one.")

    # Initialize SentencePiece processor if a model path is provided
    tokenizer_obj = None
    if spm_model_path:
        tokenizer_obj = spm.SentencePieceProcessor()
        tokenizer_obj.load(spm_model_path)

    if sub_shard_dirs:
        shard_paths = []
        for sub_dir in sub_shard_dirs:
            shard_paths.extend(
                sorted(glob.glob(os.path.join(data_dir, sub_dir, "shard-*.tar"))))
    else:
        shard_paths = sorted(glob.glob(os.path.join(data_dir, "shard-*.tar")))

    # Define the pipeline: load -> decode -> structure -> preprocess
    dataset = (
        wds.WebDataset(shard_paths)
        .decode(wds.torch_audio)
        .to_tuple("audio", "text", "meta")
        .map(lambda x: {"audio": x[0], "text": x[1], "meta": x[2]})
        .map(lambda x: preprocess_sample(x, do_tokenization, tokenizer_obj))
    )
    if max_sample_length:
        dataset = dataset.select(
            lambda x: len(x["input_values"]) <= max_sample_length)
    if dynamic_batch is not None:
        dataset = dataset.compose(
            lambda samples: _dynamic_length_batched_stream(
                samples, dynamic_batch))
    elif batch_size and length_bucket_window_mult > 1:
        dataset = dataset.compose(
            lambda samples: _length_bucketed_stream(
                samples, batch_size, length_bucket_window_mult))
    return dataset
