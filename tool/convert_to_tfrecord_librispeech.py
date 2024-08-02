#!/usr/bin/python3

# Standard imports
import uuid
import re

# import third-party util
import os
import soundfile as sf
import tensorflow as tf


def _num_examples(transcript_file):
    """Counts the number of utterances from the transcript file.

     This routine assumes that the number of lines (possibly except empty
     lines) is the number of utterances stored in TFRecords files.

     Args:
         transcript_file: A text file storing the transcript information.

     Returns:
         None.
     """
    number_of_utterances = 0
    for line in _read_line_generator(transcript_file):
        # Processes non-empty lines.
        if line.strip():
            number_of_utterances += 1
    return number_of_utterances


def _read_line_generator(file_name):
    """Generator reading each line of the file.

    Args:
     file_name: A file to be read.

    Returns:
     Generator for each line of the file.
    """
    with open(file_name) as f:
        line = True
        while line:
            line = f.readline()
            yield line.rstrip()


# Custom imports
from data.format import speech_data_pb2


# TODO(chanw.com)
#
# Refactors this with the original convert_to_tfrecord_librispeech
def process(wav_top_dir, out_tfrecord_top_dir, trans_fn, out_tfrecord_fn,
            num_shards):
    num_examples = _num_examples(trans_fn)
    examples_per_shard = num_examples / num_shards
    prev_shard_index = 0

    print(num_examples)

    with open(trans_fn, "rt") as file:
        # Crates a WaveHeader object.
        wave_header = speech_data_pb2.WaveHeader()
        wave_header.number_of_channels = 1
        wave_header.sampling_rate_hz = 16000.0
        wave_header.atomic_type = wave_header.AtomicType.INT16

        # Opens a TFRecord file.
        tfrecord_file_name = os.path.join(out_tfrecord_top_dir,
                                          out_tfrecord_fn)
        num_shards = 10
        shard_name = "{0}-00000-{1:05d}".format(tfrecord_file_name, num_shards)
        writer = tf.io.TFRecordWriter(
            shard_name, options=tf.io.TFRecordOptions(compression_type="GZIP"))

        example_index = 0

        line = True
        while line:
            line = file.readline().rstrip()
            if line:
                # Creates a SpeechData object.
                speech_data = speech_data_pb2.SpeechData()

                result = re.match(r"\[(\S*)\]\s*(.*)", line)

                wave_fn = result.group(1)
                transcript = result.group(2).strip()
                wave_fn = os.path.join(wav_top_dir, wave_fn)

                # Opens the wave data.
                with sf.SoundFile(wave_fn) as sf_file:
                    data = sf_file.read(dtype="int16")

                example_index += 1
                shard_index = min(int(example_index // examples_per_shard),
                                  num_shards - 1)

                # Opens a new shard if shard_index has been changed.
                #
                # Closes the current TFRecord shard file, and opens a new TFRecord
                # shard file.
                if shard_index != prev_shard_index:
                    writer.close()

                    # Makes the corresponding shard name.
                    shard_name = "{0}-{1:05d}-{2:05d}".format(
                        tfrecord_file_name, shard_index, num_shards)

                    writer = tf.io.TFRecordWriter(
                        shard_name,
                        options=tf.io.TFRecordOptions(compression_type="GZIP"))

                    prev_shard_index = shard_index

                speech_data.utterance_id = uuid.uuid4().hex
                speech_data.wave_header.CopyFrom(wave_header)
                speech_data.samples = data.tobytes()
                speech_data.transcript = transcript

                # TODO(chanwcom)
                # Think about how to handel the map field in the general case.

                writer.write(speech_data.SerializeToString())

        writer.close()


WAV_TOP_DIR = "/mnt/nas2dual/database/libri_speech/org_decompressed"
TRANS_TOP_DIR = "/mnt/nas2dual/database/libri_speech/tool"
OUT_TOP_DIR = "/mnt/nas2dual/database/libri_speech/tfrecord"
subset_list = [
    "test-clean",
    "test-other",
    "dev-clean",
    "dev-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
]

for subset in subset_list:
    trans_fn = os.path.join(TRANS_TOP_DIR, subset)
    trans_fn = f"{trans_fn}.trans.txt"
    out_tfrecord_fn = os.path.join(OUT_TOP_DIR, subset)
    out_tfrecord_fn = f"{out_tfrecord_fn}.tfrecord"
    process(WAV_TOP_DIR, OUT_TOP_DIR, trans_fn, out_tfrecord_fn, 10)
