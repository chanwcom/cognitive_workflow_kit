#!usr/bin/python3
""Convert Stop dataset into WebDataset format.

This script scans a LibriSpeech split (e.g., train-clean-100) and creates
.tar shard files containing FLAC audio and transcripts using the WebDataset
format.

Example usage:   
    python create_stop_webdataset.py \
        --dataset_dir 
        --output_dir 
        --shard_size_gb 

    # Convert dev-clean with smaller shard size
    python create_stop_webdataset.py \
        --dataset_dir 
        --output_dir 
        --shard_size_gb 


    # Batch process splits
    for split in train-clean-100 train-clean-360 train-other-500; do
          python create_stop_webdataset.py \
        --dataset_dir 
        --output_dir 
        --shard_size_gb 
    done

    # Debug mode with very small shard
    python create_librispeech_webdataset.py \
        --dataset_dir ./LibriSpeech/dev-clean \
        --output_dir ./wds/dev-clean-debug \
        --shard_size_gb 0.1
"""

import argparse
import os
import json
import jsonschema
import re
import uuid
import webdataset as wds

from jsonschema import validate, ValidationError
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

BYTES_PER_GB = 1024**3

def estimate_total_size(data_pairs):
    """Estimates total size of all samples in bytes.

    Args:
        data_pairs (list): List of (flac_path, meta) tuples.

    Returns:
        int: Total estimated size in bytes.
    """
    total_size = 0

    for flac_path, meta in tqdm(data_pairs, desc="Estimating total size"):
        total_size += os.path.getsize(flac_path)
        total_size += len(json.dumps(meta, ensure_ascii=False).encode("utf-8"))

    return total_size

def load_json_schema(schema_path):
    """Load and return JSON schema from a file.

    Args:
        schema_path (str): Path to the JSON schema file.

    Returns:
        dict: Parsed JSON schema as a Python dictionary.
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize_intent_string(intent_str):
    """Tokenize the intent string into bracket tokens and words.

    Args:
        intent_str (str): Intent string formatted with brackets.

    Returns:
        list[str]: List of tokens (brackets and words).
    """
    return re.findall(r'\[|\]|[^\s\[\]]+', intent_str)


def parse_tokens(tokens):
    """Parse token list recursively into a nested intent dictionary.

    Args:
        tokens (list[str]): List of tokens to parse.

    Returns:
        dict or str: Parsed intent dictionary or a string token.
    """
    if not tokens:
        return None

    token = tokens.pop(0)

    if token == "[":
        label = tokens.pop(0)
        if label.startswith("IN:"):
            intent_name = label[3:]
            elements = []
            while tokens[0] != "]":
                elements.append(parse_tokens(tokens))
            tokens.pop(0)  # Remove closing ']'
            slots = [e for e in elements if isinstance(e, dict)]
            return {"intent": intent_name, "slots": slots}

        if label.startswith("SL:"):
            slot_name = label[3:]
            value = parse_tokens(tokens)
            tokens.pop(0)  # Remove closing ']'
            if isinstance(value, dict):
                return {"slot_name": slot_name, "intent": value}
            return {"slot_name": slot_name, "slot_value": value}

    return token


def parse_intent_string(intent_str):
    """Parse an intent string into a list of intent dictionaries.

    Args:
        intent_str (str): Intent string with nested bracket notation.

    Returns:
        list[dict]: List containing parsed intent dictionary.
    """
    tokens = tokenize_intent_string(intent_str)
    parsed = parse_tokens(tokens)
    return [parsed]


def find_audio_meta_pairs_tsv_with_schema(root_dir, schema, parse_intent_string):
    """Find (audio_path, meta_dict) pairs from TSV and parse intents to JSON.

    This function reads TSV files under root_dir. Each line must have at least
    6 tab-separated fields:
      audio_filename<TAB>gender<TAB>native_flag<TAB>domain<TAB>text<TAB>intent_string

    It parses intent_string into JSON using `parse_intent_string` and validates
    the result with the provided JSON schema.

    Args:
        root_dir (str or Path): Directory containing manifest TSV and audio files.
        schema (dict): JSON schema dict to validate the meta data.
        parse_intent_string (func): Function to parse intent string to JSON.

    Returns:
        list: List of tuples (audio_path: Path, meta_dict: dict).
              meta_dict has keys: gender, native_english_flag, intents (JSON).
    """
    root_dir = Path(root_dir)
    pairs = []

    manifest_files = list(root_dir.glob("*.tsv"))

    for manifest_file in manifest_files:
        with open(manifest_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                fields = line.split("\t")
                if len(fields) < 6:
                    print(f"Skipping malformed line {line_num} in {manifest_file}")
                    continue

                audio_filename, gender, native_flag, domain, text, intent_str = fields
                audio_path = root_dir / audio_filename

                try:
                    intents_json = parse_intent_string(intent_str)
                except Exception as e:
                    print(f"Intent parse error at line {line_num} in {manifest_file}: {e}")
                    continue

                meta = {
                    "gender": gender,
                    "native_english_flag": native_flag,
                    "intents": intents_json,
                }

                try:
                    jsonschema.validate(instance=meta, schema=schema)
                except jsonschema.ValidationError as e:
                    print(f"Schema validation error at line {line_num} in {manifest_file}: {e.message}")
                    continue

                pairs.append((audio_path, meta))

    return pairs



def create_webdataset(audio_dir, output_dir, schema_path, shard_size_gb=1.0):
    """Create a WebDataset tar archive from manifest and audio files with sharding.

    This function reads a manifest TSV file with lines containing audio
    filename, speaker gender, native flag, domain, text, and intent string.
    It parses the intent string into JSON, validates it against the
    provided JSON schema, and writes the audio, text, and meta info as
    samples into sharded WebDataset tar files.

    Args:
        manifest_path (str): Path to the manifest TSV file.
        audio_dir (str): Directory containing audio files (.wav).
        output_dir (str): Directory to save shard tar files.
        schema_path (str): Path to the JSON schema file for validation.
        shard_size_gb (float): Maximum shard size in gigabytes. Default is 1GB.

    Returns:
        None
    """
    schema = load_json_schema(schema_path)
    os.makedirs(output_dir, exist_ok=True)

    current_size = 0
    sink = None

    # Find manifest.tsv under audio_dir
    audio_dir_path = Path(audio_dir)
    tsv_files = list(audio_dir_path.glob("*.tsv"))
    if len(tsv_files) != 1:
        raise FileNotFoundError(f"Expected exactly one .tsv file in {audio_dir_path}, found {len(tsv_files)}")
    manifest_path = str(tsv_files[0])

    shard_path = os.path.join(output_dir, "shard-%06d.tar")

    sink = wds.ShardWriter(shard_path, maxsize=shard_size_gb * BYTES_PER_GB)

    with open(manifest_path, "r", encoding="utf-8") as file_:
        for line in file_:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                print(f"Skipping malformed line: {line.strip()}")
                continue

            # The 4th field is skipped since all files are in the music domain.
            # TODO(chanwcom) Fix it later to handle other domains as well.
            wav_name, gender, native_flag, _, text, intent_str = parts
            wav_path = os.path.join(audio_dir, wav_name)
            base_key = os.path.splitext(wav_name)[0]

            try:
                intents = parse_intent_string(intent_str)
            except Exception as err:
                print(f"Intent parse error at {wav_name}: {err}")
                continue

            meta = {
                "gender": gender,
                "native_english_flag": native_flag,
                "intents": intents,
            }

            try:
                validate(instance=meta, schema=schema)
            except ValidationError as err:
                print(f"Schema validation error at {wav_name}: {err.message}")
                continue

            try:
                audio_bytes = open(wav_path, "rb").read()
                meta_json_str = json.dumps(meta, ensure_ascii=False)
                sample = {
                    "__key__": base_key,
                    "wav": audio_bytes,
                    "txt": text,
                    "json": meta_json_str,
                }

                sink.write(sample)

            except Exception as err:
                print(f"Error processing sample {wav_name}: {err}")

    if sink is not None:
        sink.close()
    print("WebDataset creation complete!")


def load_json_schema(schema_path):
    """Load and return JSON schema from a file.

    Args:
        schema_path (str): Path to the JSON schema file.

    Returns:
        dict: Parsed JSON schema as a Python dictionary.
    """

    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """Main entry point: parses arguments and creates shards."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Path to the data directory."
    )
    parser.add_argument(
        "--schema_file",
        type=str,
        required=True,
        help="A schema file name."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for WebDataset shards"
    )
    parser.add_argument(
        "--shard_size_gb",
        type=float,
        default=1.0,
        help="Maximum shard size in gigabytes (default: 1.0)"
    )
    parser.add_argument(
        "--min_shard_count",
        type=int,
        default=0,
        help="Minimum number of shards to generate. Overrides shard_size_gb "
             "if necessary."
    )
    args = parser.parse_args()

    data_pairs = find_audio_meta_pairs_tsv_with_schema(
        args.audio_dir, load_json_schema(args.schema_file), parse_intent_string)
    print(f"Found {len(data_pairs)} audio-transcript pairs.")

    effective_shard_size_gb = args.shard_size_gb
    if args.min_shard_count > 0:
        total_size_bytes = estimate_total_size(data_pairs)
        # TODO(chanwcom) 2% margin is added.
        # TODO(chanwcom) Remove this hack.
        min_shard_gb = total_size_bytes / args.min_shard_count / BYTES_PER_GB  * 1.01
        if min_shard_gb < args.shard_size_gb:
            print(f"Adjusting shard size from {args.shard_size_gb:.3f} GB to "
                  f"{min_shard_gb:.3f} GB to satisfy min_shard_count="
                  f"{args.min_shard_count}")
            effective_shard_size_gb = min_shard_gb

    create_webdataset(
        args.audio_dir, args.output_dir, args.schema_file, effective_shard_size_gb)

if __name__ == "__main__":
    main()
