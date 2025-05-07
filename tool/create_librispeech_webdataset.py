import os
import argparse
import tarfile
import uuid
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import webdataset as wds

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
        # LibriSpeech transcript files are per-chapter, not per-utterance.
        transcript_path = flac_path.with_suffix(".txt")
        if not transcript_path.exists():
            # Construct transcript file path from speaker and chapter IDs.
            transcript_dir = flac_path.parent
            speaker_id = flac_path.parts[-3]
            chapter_id = flac_path.parts[-2]
            transcript_file = transcript_dir / f"{speaker_id}-{chapter_id}.trans.txt"
            if not transcript_file.exists():
                continue

            # Load all utterance transcripts from the file into a dictionary.
            with open(transcript_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            trans_dict = {
                line.split(" ", 1)[0]: line.split(" ", 1)[1].strip() for line in lines
            }

            uid = flac_path.stem  # e.g., "84-121123-0001"
            if uid in trans_dict:
                data_pairs.append((flac_path, trans_dict[uid]))

    return data_pairs

def write_shards(data_pairs, output_dir, shard_size):
    """Writes (FLAC, transcript) pairs into WebDataset shards.

    Args:
        data_pairs (list): List of (flac_path, transcript) tuples.
        output_dir (str): Path to directory where shards are written.
        shard_size (int): Maximum number of samples per shard.
    """
    os.makedirs(output_dir, exist_ok=True)
    shard_id = 0
    sink = None

    for idx, (flac_path, transcript) in enumerate(
            tqdm(data_pairs, desc="Writing shards")):
        # Start a new shard if current one reached max size.
        if idx % shard_size == 0:
            if sink is not None:
                sink.close()
            shard_path = os.path.join(output_dir, f"shard-{shard_id:06d}.tar")
            sink = wds.ShardWriter(shard_path, maxcount=shard_size)
            shard_id += 1

        # Read audio as raw bytes to store in WebDataset.
        with open(flac_path, "rb") as f:
            audio_bytes = f.read()

        # Create a unique key for this sample.
        sample_key = str(uuid.uuid4())

        sample = {
            "__key__": sample_key,   # Required key for WebDataset
            "flac": audio_bytes,     # Audio data in FLAC format
            "txt": transcript,       # Raw transcript as string
        }

        sink.write(sample)

    # Close the last shard writer.
    if sink is not None:
        sink.close()

def main():
    """Main entry point: parses arguments and creates shards."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to LibriSpeech split directory (e.g., ./LibriSpeech/train-clean-100)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for WebDataset shards"
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=1000,
        help="Number of samples per shard (default: 1000)"
    )
    args = parser.parse_args()

    # Scan directory for audio-transcript pairs.
    data_pairs = find_audio_transcript_pairs(args.dataset_dir)
    print(f"Found {len(data_pairs)} audio-transcript pairs.")

    # Write to WebDataset shards.
    write_shards(data_pairs, args.output_dir, args.shard_size)

if __name__ == "__main__":
    main()

