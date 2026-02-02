# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard library imports
import glob
import io
import os
from typing import Dict, Any, Optional, Union

# Third-party imports
import sentencepiece as spm
import torchaudio
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
    # Hack
    # TODO(chanwcom)
    # Need to make the following changes:
    # 1. Prepends and appends <s> and </s> only when those are required as options.
    # 2. Even if this option is enabled, check whether <s> and </s> are already
    #     there, and in this case, do not pre-pend or append.
    text = f"<s> {text} </s>"
    if isinstance(text, bytes):
        text = text.decode("utf-8").strip()
    else:
        text = text.strip()


    # Tokenize text using the provided tokenizer or default processor
    if do_tokenization:
        if tokenizer_obj is None:
            labels = processor.tokenizer(text).input_ids
        elif isinstance(tokenizer_obj, spm.SentencePieceProcessor):
            # SentencePiece encoding returns a list of integers
            labels = tokenizer_obj.encode(text, out_type=int)
        else:
            labels = tokenizer_obj(text).input_ids
    else:
        labels = text

    return {"input_values": input_values, "labels": labels}


def make_dataset(
    data_dir: str,
    do_tokenization: bool = True,
    spm_model_path: Optional[str] = None
) -> wds.WebDataset:
    """Create a WebDataset pipeline with optional SentencePiece support.
    It reads all shards named 'shard-*.tar' in the given directory,
    extracts 'wav' and 'txt' entries as tuples, converts them into dictionaries,
    and applies the preprocessing function.


    Args:
        data_dir: Path to directory containing 'shard-*.tar'.
        do_tokenization: Whether to apply tokenization during mapping.
        spm_model_path: Path to the SentencePiece *.model file.

    Returns:
        A prepared WebDataset pipeline.
    """
    # Initialize SentencePiece processor if a model path is provided
    tokenizer_obj = None

    if spm_model_path:
        tokenizer_obj = spm.SentencePieceProcessor()
        tokenizer_obj.load(spm_model_path)

    # Define the pipeline: load -> decode -> structure -> preprocess
    dataset = (
        wds.WebDataset(glob.glob(os.path.join(data_dir, "shard-*.tar")))
        .decode(wds.torch_audio)
        .to_tuple("audio", "text", "meta")
        .map(lambda x: {"audio": x[0], "text": x[1], "meta": x[2]})
        .map(lambda x: preprocess_sample(x, do_tokenization, tokenizer_obj))
    )
    return dataset
