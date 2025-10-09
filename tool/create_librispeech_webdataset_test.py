import shutil
import tempfile
import unittest
from pathlib import Path

from create_librispeech_webdataset import (
    find_audio_transcript_pairs,
    estimate_total_size,
    write_shards,
)


class TestCreateLibriSpeechWebDataset(unittest.TestCase):

    def setUp(self):
        """Set up a temporary LibriSpeech-like structure with one audio-text pair."""
        self.temp_dir = tempfile.mkdtemp()
        speaker = "123"
        chapter = "456"
        utt_id = f"{speaker}-{chapter}-0001"
        self.utt_dir = Path(self.temp_dir) / speaker / chapter
        self.utt_dir.mkdir(parents=True, exist_ok=True)

        # Write dummy FLAC file
        self.flac_path = self.utt_dir / f"{utt_id}.flac"
        self.flac_path.write_bytes(b"FAKEFLACDATA")

        # Write transcript file
        self.transcript_file = self.utt_dir / f"{speaker}-{chapter}.trans.txt"
        self.transcript_file.write_text(f"{utt_id} Hello world\n")

    def tearDown(self):
        """Clean up temporary files after each test."""
        shutil.rmtree(self.temp_dir)

    def test_find_audio_transcript_pairs(self):
        pairs = find_audio_transcript_pairs(self.temp_dir)
        self.assertEqual(len(pairs), 1)
        flac, text = pairs[0]
        self.assertTrue(flac.name.endswith(".flac"))
        self.assertEqual(text, "Hello world")

    def test_estimate_total_size(self):
        pairs = find_audio_transcript_pairs(self.temp_dir)
        total_size = estimate_total_size(pairs)
        expected_size = len(b"FAKEFLACDATA") + len(
            "Hello world".encode("utf-8"))
        self.assertEqual(total_size, expected_size)

    def test_write_shards(self):
        pairs = find_audio_transcript_pairs(self.temp_dir)
        output_dir = tempfile.mkdtemp()

        try:
            write_shards(pairs, output_dir,
                         shard_size_gb=0.00001)  # very small to force split
            shard_files = list(Path(output_dir).glob("shard-*.tar"))
            self.assertGreaterEqual(len(shard_files), 1)
            for shard_file in shard_files:
                self.assertGreater(shard_file.stat().st_size, 0)
        finally:
            shutil.rmtree(output_dir)


if __name__ == '__main__':
    unittest.main()
