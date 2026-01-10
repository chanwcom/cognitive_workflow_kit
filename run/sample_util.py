# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard library imports
import glob
import io
import os
from typing import Dict

# Third-party imports
import torchaudio
import webdataset as wds
from transformers import AutoProcessor

# Define processor globally (assumed to be initialized elsewhere in actual code)
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

def preprocess_sample(sample: Dict, do_tokenization: bool=True) -> Dict:
    """Preprocess a single raw sample from the WebDataset.

    This function loads the waveform from the raw bytes using torchaudio,
    extracts features using the processor's feature extractor, and tokenizes
    the transcript text.

    Args:
        sample (Dict): A dictionary containing keys 'wav' (raw audio bytes)
            and 'txt' (transcript bytes).

    Returns:
        Dict: A dictionary with keys:
            - 'input_values': processed audio feature tensor.
            - 'labels': list of token IDs corresponding to the transcript.
    """
    waveform, sample_rate = torchaudio.load(io.BytesIO(sample["audio"]))
    input_values = processor.feature_extractor(
        waveform[0], sampling_rate=sample_rate
    ).input_values[0]

    # Processes text to handle both bytes and a string.
    if isinstance(sample["text"], bytes):
        text = sample["text"].decode("utf-8").strip()
    else:
        text = sample["text"].strip()

    if do_tokenization:
        labels = processor.tokenizer(text).input_ids
    else:
        labels = text

    return {"input_values": input_values, "labels": labels}


def make_dataset(data_dir: str, do_tokenization: bool=True) -> wds.WebDataset:
    """Create a WebDataset pipeline that loads and preprocesses data shards.

    It reads all shards named 'shard-*.tar' in the given directory,
    extracts 'wav' and 'txt' entries as tuples, converts them into dictionaries,
    and applies the preprocessing function.

    Args:
        data_dir (str): Path to the directory containing dataset shards.

    Returns:
        wds.WebDataset: The prepared dataset pipeline with preprocessing.
    """
    dataset = (
        wds.WebDataset(glob.glob(os.path.join(data_dir, "shard-*.tar")))
            .decode(wds.torch_audio)
            .to_tuple("audio", "text", "meta")
            .map(lambda sample: {"audio": sample[0], "text": sample[1], "meta": sample[2]})
            .map(lambda inputs: preprocess_sample(inputs, do_tokenization))
    )
    return dataset
