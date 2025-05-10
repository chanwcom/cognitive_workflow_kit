"""Converts the LibriSpeech dataset into WebDataset format.

This script scans a LibriSpeech split (e.g., train-clean-100) and creates
.tar shard files containing FLAC audio and transcripts using the WebDataset
format.

Example usage:
    # Converts train-clean-100 with default shard size (in GB)
    python create_librispeech_webdataset.py \
        --dataset_dir ./LibriSpeech/train-clean-100 \
        --output_dir ./wds/train-clean-100 \
        --shard_size_gb 1.0

    # Converts dev-clean with smaller shard size
    python create_librispeech_webdataset.py \
        --dataset_dir ./LibriSpeech/dev-clean \
        --output_dir ./wds/dev-clean \
        --shard_size_gb 0.5

    # Batch process splits
    for split in train-clean-100 train-clean-360 train-other-500; do
        python create_librispeech_webdataset.py \
            --dataset_dir ./LibriSpeech/$split \
            --output_dir ./wds/$split \
            --shard_size_gb 1.5
    done

    # Debug mode with very small shard
    python create_librispeech_webdataset.py \
        --dataset_dir ./LibriSpeech/dev-clean \
        --output_dir ./wds/dev-clean-debug \
        --shard_size_gb 0.1
"""

import os
import argparse
import uuid
from pathlib import Path
from tqdm import tqdm
import webdataset as wds

BYTES_PER_GB = 1 << 30

def find_audio_transcript_pairs(root_dir):
    """Finds all (FLAC, transcript) pairs in a LibriSpeech-style directory.

    Args:
        root_dir (str or Path): Root directory of a LibriSpeech split.

    Returns:
        list: A list of tuples (flac_path, transcript_string).
    """
    flac_paths = list(Path(root_dir).rglob("*.flac"))
    data_pairs = []

    for flac_path in flac_paths:
        transcript_path = flac_path.with_suffix(".txt")
        if not transcript_path.exists():
            transcript_dir = flac_path.parent
            speaker_id = flac_path.parts[-3]
            chapter_id = flac_path.parts[-2]
            transcript_file = transcript_dir / f"{speaker_id}-{chapter_id}.trans.txt"
            if not transcript_file.exists():
                continue

            with open(transcript_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            trans_dict = {
                line.split(" ", 1)[0]: line.split(" ", 1)[1].strip()
                for line in lines
            }

            uid = flac_path.stem
            if uid in trans_dict:
                data_pairs.append((flac_path, trans_dict[uid]))

    return data_pairs

def estimate_total_size(data_pairs):
    """Estimates total size of all samples in bytes.

    Args:
        data_pairs (list): List of (flac_path, transcript) tuples.

    Returns:
        int: Total estimated size in bytes.
    """
    total_size = 0
    for flac_path, transcript in tqdm(data_pairs, desc="Estimating total size"):
        total_size += os.path.getsize(flac_path)
        total_size += len(transcript.encode("utf-8"))
    return total_size

def write_shards(data_pairs, output_dir, shard_size_gb):
    """Writes (FLAC, transcript) pairs into WebDataset shards.

    Args:
        data_pairs (list): List of (flac_path, transcript) tuples.
        output_dir (str): Path to directory where shards are written.
        shard_size_gb (float): Maximum shard size in gigabytes.
    """
    # Ensures the output directory exists, create if it doesn't.
    os.makedirs(output_dir, exist_ok=True)

    # Initializes variables for shard ID and current shard size tracking.
    shard_id = 0
    current_size = 0
    sink = None

    # Defines the shard path format and initialize the ShardWriter.
    shard_path = os.path.join(output_dir, f"shard-%06d.tar")
    sink = wds.ShardWriter(shard_path, maxsize=shard_size_gb * (1 << 30))

    # Iterates through all data pairs (FLAC file path and transcript).
    for idx, (flac_path, transcript) in enumerate(
            tqdm(data_pairs, desc="Writing shards")):

        # Reads the FLAC audio file into memory.
        with open(flac_path, "rb") as f:
            audio_bytes = f.read()

        # Generates a unique key for the sample.
        sample_key = str(uuid.uuid4())

        # Creates a sample dictionary with the audio and transcript.
        sample = {
            "__key__": sample_key,
            "flac": audio_bytes,
            "txt": transcript,
        }

        # Writes the sample to the current shard.
        sink.write(sample)

    # Closes the ShardWriter after processing all the samples.
    if sink is not None:
        sink.close()

def main():
    """Main entry point: parses arguments and creates shards."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to LibriSpeech split directory "
             "(e.g., ./LibriSpeech/train-clean-100)"
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
        default=10,
        help="Minimum number of shards to generate. Overrides shard_size_gb "
             "if necessary."
    )
    args = parser.parse_args()

    data_pairs = find_audio_transcript_pairs(args.dataset_dir)
    print(f"Found {len(data_pairs)} audio-transcript pairs.")

    effective_shard_size_gb = args.shard_size_gb
    if args.min_shard_count > 0:
        total_size_bytes = estimate_total_size(data_pairs)
        min_shard_gb = total_size_bytes / args.min_shard_count / BYTES_PER_GB
        if min_shard_gb < args.shard_size_gb:
            print(f"Adjusting shard size from {args.shard_size_gb:.3f} GB to "
                  f"{min_shard_gb:.3f} GB to satisfy min_shard_count="
                  f"{args.min_shard_count}")
            effective_shard_size_gb = min_shard_gb

    write_shards(data_pairs, args.output_dir, effective_shard_size_gb)

if __name__ == "__main__":
    main()
