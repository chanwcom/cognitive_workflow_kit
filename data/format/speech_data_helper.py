"""A module implementing helpder methods for the SpeechData proto-message."""

# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import uuid

# Third-party imports
import tensorflow as tf
from google.protobuf import descriptor_pb2
from packaging import version

# Custom imports
from data.format import speech_data_pb2
from operation import operation

assert version.parse(tf.__version__) >= version.parse("2.0.0"), (
    "At least tensorflow 2.0 is required.")


class WaveToSpeechData(operation.AbstractOperation):
    """"""

    def __init__(self):
        pass

    def process(self, wave_data_list):
        """Converts a list of wave data into a list of SpeechData.

        Args:
            inputs: A tuple of the following format:
             ({"WAVE_HEADER": A list of  ,
              "SEQ_DATA"
              "SEQ_LEN"


                "SEQ_DATA": a batch of waveform data,
               "SEQ_LEN": a batch of lengths of waveform data},
              {"SEQ_DATA": a batch of label data,
               "SEQ_LEN": a batch of length of label data. It should be always one.})

        Returns:
            A list of SpeechData.
        """
        assert isinstance(wave_data_list, list)

        speech_data_list = []
        for item in wave_data_list:
            (wave_header, waveform_data, transcript) = item

            # Sets the UtteranceData protocol buffer.
            speech_data = speech_data_pb2.SpeechData()

            speech_data.utterance_id = uuid.uuid4().hex
            speech_data.wave_header.CopyFrom(wave_header)
            speech_data.samples = waveform_data.tobytes()
            speech_data.transcript = transcript
            speech_data_list.append(speech_data)

        return speech_data_list


class SpeechDataToWave(operation.AbstractOperation):

    def __init__(self, output_type):
        self._output_type = output_type

        # This serialized string is needed for "tf.io.decode_proto".
        # We use tf.io.decode_proto instead of ParseFromString for efficiency
        # reason. Obtains the serialized descriptor information.
        file_descriptor_set = descriptor_pb2.FileDescriptorSet()
        file_descriptor = file_descriptor_set.file.add()
        speech_data_pb2.DESCRIPTOR.CopyToProto(file_descriptor)
        self._string_descriptor = (b"bytes://" +
                                   file_descriptor_set.SerializeToString())

    def process(self, inputs):
        acoust_seq_data = []
        acoust_seq_len = []
        acoust_max_seq_len = 0
        acoust_sampling_rate_hz = []

        label_seq_data = []
        label_seq_len = []

        # TODO(chanwcom) Check whether we can use vectorized map instead.
        for i in range(tf.shape(inputs)[0]):
            outputs = parse_speech_data(inputs[i], self._string_descriptor,
                                        self._output_type)
            acoust_seq_data.append(outputs[0]["SEQ_DATA"])
            acoust_seq_len.append(outputs[0]["SEQ_LEN"])
            acoust_max_seq_len = tf.math.reduce_max(
                (acoust_max_seq_len, outputs[0]["SEQ_LEN"]))
            acoust_sampling_rate_hz.append(outputs[0]["SAMPLING_RATE_HZ"])

            label_seq_data.append(outputs[1]["SEQ_DATA"])
            label_seq_len.append(outputs[1]["SEQ_LEN"])

        for i in range(tf.shape(inputs)[0]):
            acoust_seq_data[i] = tf.pad(
                acoust_seq_data[i],
                [[0, acoust_max_seq_len - acoust_seq_len[i]], [0, 0]])

        acoust_dict = {}
        acoust_dict["SEQ_LEN"] = tf.convert_to_tensor(acoust_seq_len)
        acoust_dict["SEQ_DATA"] = tf.convert_to_tensor(acoust_seq_data)
        acoust_dict["SAMPLING_RATE_HZ"] = tf.convert_to_tensor(
            acoust_sampling_rate_hz)

        label_dict = {}
        label_dict["SEQ_DATA"] = tf.convert_to_tensor(label_seq_data)
        label_dict["SEQ_LEN"] = tf.convert_to_tensor(label_seq_len)

        return (acoust_dict, label_dict)


def parse_speech_data(speech_data_string,
                      string_descriptor,
                      out_type=tf.dtypes.float32):
    """Parses a single example serialized in SpeechData proto-message.

    Args:
        speech_data_string:
        string_descriptor:
        out_type:

    Returns:

    """
    # Parses speech_data_string containing learning.SpeechData proto-message.
    speech_data_parsed = tf.io.decode_proto(
        speech_data_string,
        "learning.SpeechData", ["wave_header", "samples", "transcript"],
        [tf.string, tf.string, tf.string],
        descriptor_source=string_descriptor)
    serialized_samples = speech_data_parsed.values[1][0]

    # Parses the wave_header using "tf.io.decode_proto".
    wave_header_parsed = tf.io.decode_proto(
        speech_data_parsed.values[0][0],
        "learning.WaveHeader",
        ["number_of_channels", "sampling_rate_hz", "atomic_type"],
        [tf.int32, tf.float64, tf.int32],
        descriptor_source=string_descriptor)

    # The following assumes that the number of channels in a given batch is
    # the same. Anyway, if there are some examples whose number of channels
    # differ, then an error will occur.
    num_channels = wave_header_parsed.values[0][0]
    atomic_type = wave_header_parsed.values[2][0]

    scaling = 1.0
    if atomic_type == speech_data_pb2.WaveHeader.INT4:
        tf.debugging.Assert(False, [tf.constant("Unsupported Type.")])
    elif atomic_type == speech_data_pb2.WaveHeader.INT8:
        dtypes = tf.dtypes.int8
        scaling = 1.0 / 2**7
    elif atomic_type == speech_data_pb2.WaveHeader.INT16:
        dtypes = tf.dtypes.int16
        scaling = 1.0 / 2**15
    elif atomic_type == speech_data_pb2.WaveHeader.INT32:
        dtypes = tf.dtypes.int32
        scaling = 1.0 / 2**31
    elif atomic_type == speech_data_pb2.WaveHeader.FLOAT16:
        dtypes = tf.dtypes.float16
    elif atomic_type == speech_data_pb2.WaveHeader.FLOAT32:
        dtypes = tf.dtypes.float32
    elif atomic_type == speech_data_pb2.WaveHeader.FLOAT64:
        dtypes = tf.dtypes.float64

    samples = tf.reshape(
        tf.cast(tf.io.decode_raw(serialized_samples, dtypes), out_type),
        (-1, num_channels))
    samples = samples * scaling

    seq_len = tf.shape(samples)[0]

    acoust_dict = {}
    acoust_dict["SEQ_LEN"] = seq_len
    acoust_dict["SEQ_DATA"] = samples
    acoust_dict["SAMPLING_RATE_HZ"] = tf.cast(wave_header_parsed.values[1][0],
                                              out_type)

    label_dict = {}
    label_dict["SEQ_DATA"] = speech_data_parsed.values[2][0]
    label_dict["SEQ_LEN"] = tf.constant(1)

    return (acoust_dict, label_dict)
