#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter dialogs from JSON files based on a given filter list and pack
(audio, text, metadata) into WebDataset format.

Each WebDataset sample contains:
  - audio (.wav)
  - text (.txt)
  - metadata (.json) with both dataset-level and dialog-level info

Example usage:
  ./filter_and_pack.py \
    --filter-list A.txt \
    --json-dir /path/to/jsons \
    --data-root /data/root \
    --output-pattern output-%06d.tar \
    --max-per-shard 5000
"""

import os
import json
from pathlib import Path
import argparse
import webdataset as wds


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
        "--json-dir",
        required=True,
        help="Directory containing JSON files with dataset/dialog information.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root directory containing actual audio/text data.",
    )
    parser.add_argument(
        "--output-pattern",
        default="output-%06d.tar",
        help="Output WebDataset shard pattern (default: output-%%06d.tar).",
    )
    parser.add_argument(
        "--max-per-shard",
        type=int,
        default=5000,
        help="Maximum number of samples per shard (default: 5000).",
    )
    return parser.parse_args()


def load_target_files(txt_path):
    """Load target file names from the filter list into a set."""
    with open(txt_path, "r", encoding="utf-8") as f:
        targets = set(line.strip() for line in f if line.strip())
    print(f"[INFO] Loaded {len(targets):,} target file names from {txt_path}.")
    return targets


def process_json(json_path, target_files, data_root, sink):
    """Process a single JSON file and write matching dialogs to WebDataset."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            js = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {json_path}: {e}")
        return

    dataset_meta = js.get("dataSet", {})
    dialogs = dataset_meta.get("dialogs", [])

    for dialog in dialogs:
        audio_path = dialog.get("audioPath")
        if not audio_path:
            continue

        fname = Path(audio_path).name
        if fname not in target_files:
            continue

        audio_full = Path(data_root) / audio_path
        text_full = Path(data_root) / dialog.get("textPath", "")

        if not audio_full.exists() or not text_full.exists():
            print(f"[SKIP] Missing file for {audio_full}")
            continue

        metadata = {
            "dataSet": {k: v for k, v in dataset_meta.items() if k != "dialogs"},
            "dialog": dialog,
        }

        key = Path(fname).stem
        try:
            with open(audio_full, "rb") as fa, open(text_full, "rb") as ft:
                sink.write({
                    "__key__": key,
                    "wav": fa.read(),
                    "txt": ft.read(),
                    "json": json.dumps(metadata, ensure_ascii=False),
                })
        except Exception as e:
            print(f"[ERROR] Failed to write {audio_full}: {e}")


def main():
    """Main entry point."""
    args = parse_args()

    # Validate inputs
    json_dir = Path(args.json_dir)
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    target_files = load_target_files(args.filter_list)

    sink = wds.ShardWriter(args.output_pattern, maxcount=args.max_per_shard)

    json_files = sorted(json_dir.glob("*.json"))
    print(f"[INFO] Found {len(json_files):,} JSON files to process.")

    for idx, json_path in enumerate(json_files, start=1):
        process_json(json_path, target_files, data_root, sink)
        if idx % 100 == 0:
            print(f"[PROGRESS] Processed {idx}/{len(json_files)} JSON files")

    sink.close()
    print("[DONE] Finished writing WebDataset shards.")


if __name__ == "__main__":
    main()

