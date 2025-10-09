# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                      unicode_literals)

"""
Filter dialogs from JSON files based on a given filter list and pack
(audio, text, metadata) into WebDataset format.

Each WebDataset sample contains:
  - audio (.wav)
  - text (.txt)
  - metadata (.json) with both dataset-level and dialog-level info

Shard size can be controlled by shard_size_gb and min_shard_count.

Example usage:
  ./filter_and_pack.py \
    --filter-list A.txt \
    --data-root /data/root \
    --output-dir /data/output \
    --shard-size-gb 1.0 \
    --min-shard-count 10
"""
__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

import os
import json
from pathlib import Path
import argparse
import webdataset as wds
import uuid
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from typing import Set

BYTES_PER_GB = 1 << 30  # 1GB


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter JSON dialogs and pack selected items into WebDataset format."
    )
    parser.add_argument(
        "--filter-list",
        required=True,
        help="Path to text file containing filenames to include (e.g., A.txt).",
    )
    parser.add_argument(
        "--label-root",
        required=True,
        help="Root directory containing JSON files and actual label data.",
    )
    parser.add_argument(
        "--audio-root",
        required=True,
        help="Root directory containing JSON raw audio data.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for WebDataset shards.",
    )
    parser.add_argument(
        "--shard-size-gb",
        type=float,
        default=1.0,
        help="Maximum shard size in gigabytes (default: 1.0).",
    )
    parser.add_argument(
        "--min-shard-count",
        type=int,
        default=10,
        help="Minimum number of shards to generate. Overrides shard_size_gb if necessary.",
    )
    return parser.parse_args()


def load_target_files(txt_path):
    """Load target file names from the filter list into a set."""
    with open(txt_path, "r", encoding="utf-8-sig") as f:
        targets = set(line.strip() for line in f if line.strip())
    print(f"[INFO] Loaded {len(targets):,} target file names from {txt_path}.")
    return targets


def find_all_json_files(data_root):
    """Recursively find all *.json files under data_root."""
    data_root = Path(data_root)
    json_files = list(data_root.rglob("*.json"))
    print(f"[INFO] Found {len(json_files):,} JSON files under {data_root}.")
    return json_files


def estimate_total_size(json_files: Set[Path],
                        target_files: Set[Path],
                        label_root: str,
                        audio_root: str):
    """Estimate total size in bytes of matching audio + text files."""
    total_size = 0

    for json_path in tqdm(json_files, desc="Estimating total size"):
        with open(json_path, "r", encoding="utf-8-sig") as f:
            js = json.load(f)

        dataset_meta = js.get("dataSet", {})
        dialogs = dataset_meta.get("dialogs", [])

        for dialog in dialogs:
            audio_path = dialog.get("audioPath")
            audio_path = str(Path(audio_path).with_suffix(''))


            # In AI Hub consulting dataset, the top directory of
            # audio_path is an non-existent directory. Thus, we
            # remove it.
            audio_path = str(Path(*Path(audio_path).parts[1:]))

            if not audio_path or audio_path not in target_files:
                continue

            audio_full = (Path(audio_root) / audio_path
                          ).with_suffix(".wav")
            text_full = (Path(label_root) / audio_path
                         ).with_suffix(".txt")

            audio_path = Path(audio_full)

            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Audio file not found or invalid: {audio_full}"
                )

            # Replaces the original file extension with '.flac'.
            flac_path = audio_path.with_suffix(".flac")

            # Reads the input audio data and its sample rate.
            data, samplerate = sf.read(audio_path)

            # Writes the data to a new file in FLAC format.
            sf.write(flac_path, data, samplerate, format="FLAC")

            # Update the total size with the new FLAC file size.
            total_size += flac_path.stat().st_size

            if not text_full.exists():
                raise FileNotFoundError(
                    f"Text file not found or invalid: {text_full}")
            total_size += text_full.stat().st_size

    return total_size


def process_json(json_path, target_files, label_root, audio_root, sink):
    """Process a single JSON file and write matching dialogs to WebDataset."""
    try:
        with open(json_path, "r", encoding="utf-8-sig") as f:
            js = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {json_path}: {e}")
        return 0

    dataset_meta = js.get("dataSet", {})
    dialogs = dataset_meta.get("dialogs", [])

    written_count = 0
    for dialog in dialogs:
        # TODO(chanw.com)
        # Refactor this portion with the same segment in estimate_total_size.
        audio_path = dialog.get("audioPath")
        audio_path = str(Path(audio_path).with_suffix(""))

        # In AI Hub consulting dataset, the top directory of
        # audio_path is an non-existent directory. Thus, we
        # remove it.
        audio_path = str(Path(*Path(audio_path).parts[1:]))

        if not audio_path or audio_path not in target_files:
            continue

        audio_full = (Path(audio_root) / audio_path
                      ).with_suffix(".wav")
        # Replaces the original file extension with '.flac'.
        audio_full = audio_full.with_suffix(".flac")

        text_full = (Path(label_root) / audio_path
                     ).with_suffix(".txt")

        if not audio_full.exists():
            raise FileNotFoundError(
                f"Audio file not found or invalid: {audio_full}"
            )

        if not text_full.exists():
            raise FileNotFoundError(
                f"Text file not found or invalid: {text_full}"
            )

        metadata = {
            "dataSet": {k: v for k, v in dataset_meta.items() if k != "dialogs"},
            "dialog": dialog,
        }

        key = str(uuid.uuid4())
        try:
            with open(audio_full, "rb") as fa, open(text_full, "rb") as ft:
                sink.write({
                    "__key__": key,
                    "wav": fa.read(),
                    "txt": ft.read(),
                    "json": json.dumps(metadata, ensure_ascii=False),
                })
            written_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to write {audio_full}: {e}")

    return written_count


def main():
    """Main entry point."""
    args = parse_args()

    label_root = Path(args.label_root)
    if not label_root.exists():
        raise FileNotFoundError(f"Data root not found: {label_root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_files = load_target_files(args.filter_list)
    json_files = find_all_json_files(label_root)

    audio_root = Path(args.audio_root)
    if not audio_root.exists():
        raise FileNotFoundError(f"Data root not found: {audio_root}")

    # Estimate shard size
    effective_shard_size_gb = args.shard_size_gb
    if args.min_shard_count > 0:
        total_size_bytes = estimate_total_size(
            json_files, target_files, label_root, audio_root)
        min_shard_gb = total_size_bytes / args.min_shard_count / BYTES_PER_GB
        if min_shard_gb < args.shard_size_gb:
            print(f"[INFO] Adjusting shard size from {args.shard_size_gb:.3f} GB to "
                  f"{min_shard_gb:.3f} GB to satisfy min_shard_count={args.min_shard_count}")
            effective_shard_size_gb = min_shard_gb

    shard_path = str(output_dir / "shard-%06d.tar")
    sink = wds.ShardWriter(shard_path, maxsize=effective_shard_size_gb * BYTES_PER_GB)

    total_written = 0
    for json_path in tqdm(json_files, desc="Processing JSON files"):
        written = process_json(json_path, target_files, label_root, audio_root,
                               sink)
        total_written += written

    sink.close()
    print(f"[DONE] Finished writing WebDataset shards. Total samples: {total_written:,}")


if __name__ == "__main__":
    main()
