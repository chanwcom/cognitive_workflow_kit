#!/usr/bin/python3

# Standard imports
import uuid
import re

# import third-party util
import os
import soundfile as sf
import tensorflow as tf

# Custom imports
from data.format import speech_data_pb2



db_top = "/home/chanwcom/speech_database/stop/stop/train/music_train"
trans_file = "/home/chanwcom/speech_database/stop/stop/train/music_train/transcript.txt"

with open(trans_file, "rt") as file:

    # Crates a WaveHeader object.
    wave_header = speech_data_pb2.WaveHeader()
    wave_header.number_of_channels = 1
    wave_header.sampling_rate_hz = 16000.0
    wave_header.atomic_type = wave_header.AtomicType.INT16

    # Opens a TFRecord file.
    topdir = "/home/chanwcom/local_repositories/cognitive_workflow_kit/tool"
    writer = tf.io.TFRecordWriter(
        os.path.join(topdir, "music_train_gzip.tfrecord"),
        options=tf.io.TFRecordOptions(compression_type="GZIP"))

    # Creates a SpeechData object.
    speech_data = speech_data_pb2.SpeechData()

    line = True
    while line:
        line = file.readline().rstrip()
        if line:
            print(line)

            result = re.match(r"\[(\S*)\]\s*(.*)", line)

            wave_fn = result.group(1)
            transcript = result.group(2)
            wave_fn = os.path.join(db_top, wave_fn)

            # Opens the wave data.
            with sf.SoundFile(wave_fn) as sf_file:
                data = sf_file.read(dtype="int16")

            speech_data.utterance_id = uuid.uuid4().hex
            speech_data.wave_header.CopyFrom(wave_header)
            speech_data.samples = data.tobytes()
            speech_data.transcript = transcript

            writer.write(speech_data.SerializeToString())

    writer.close()
