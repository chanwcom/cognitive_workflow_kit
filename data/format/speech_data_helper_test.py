"""A module for unit-testing the "speech_data_helper" module."""

# pylint: disable=invalid-name, no-member, import-error
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import uuid

# Third-party imports
import numpy as np
import tensorflow as tf
from google.protobuf import descriptor_pb2

# Custom imports
from data.format import speech_data_helper
from data.format import speech_data_pb2


class WaveToSpeechDataTest(tf.test.TestCase):
    """A class for testing methods in the example module."""

    @classmethod
    def setUpClass(cls):
        cls._wave_data_list = []

        wave_header = speech_data_pb2.WaveHeader()
        wave_header.number_of_channels = 2
        wave_header.sampling_rate_hz = 16000.0
        wave_header.atomic_type = speech_data_pb2.WaveHeader.FLOAT32
        data = np.array([[2.0, 1.0], [3.0, 4.0], [5.0, 3.0], [1.0, 0.0]])
        transcript = "HELLO"

        cls._wave_data_list.append((wave_header, data, transcript))

        wave_header.number_of_channels = 1
        wave_header.sampling_rate_hz = 16000.0
        wave_header.atomic_type = speech_data_pb2.WaveHeader.INT16
        data = np.array([[2], [3], [5], [1]], dtype=np.int16)
        transcript = "MY"

        cls._wave_data_list.append((wave_header, data, transcript))

    def test_process_single_input(self):
        """This method tests the "process" method of WaveToSpeechData."""
        op = speech_data_helper.WaveToSpeechData()

        actual_list = op.process(self._wave_data_list)

        # Checks whether the length is two.

        self.assertEqual(self._wave_data_list[0][0],
                         actual_list[0].wave_header)
        self.assertTrue(isinstance(actual_list[0].utterance_id, str))

        try:
            np.testing.assert_array_equal(
                self._wave_data_list[0][1],
                np.reshape(np.frombuffer(actual_list[0].samples, dtype=float),
                           (-1, 2)))
        except AssertionError:
            self.fail()
        self.assertEqual("HELLO", actual_list[0].transcript)

        self.assertEqual(self._wave_data_list[1][0],
                         actual_list[1].wave_header)
        self.assertTrue(isinstance(actual_list[1].utterance_id, str))
        self.assertEqual("MY", actual_list[1].transcript)
        try:
            np.testing.assert_array_equal(
                self._wave_data_list[1][1],
                np.reshape(
                    np.frombuffer(actual_list[1].samples, dtype=np.int16),
                    (-1, 1)))
        except AssertionError:
            self.fail()


class SpeechDataToTensorTest(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        # Prepare a batch of SpeechData.

        # Make the sctring descriptor
        # Creates a serialized FileDescriptorSet.
        #
        # This serialized string is needed for "tf.io.decode_proto".
        # We use tf.io.decode_proto instead of ParseFromString for efficiency
        # reason. Obtains the serialized descriptor information.
        file_descriptor_set = descriptor_pb2.FileDescriptorSet()
        file_descriptor = file_descriptor_set.file.add()
        speech_data_pb2.DESCRIPTOR.CopyToProto(file_descriptor)
        cls._string_descriptor = (b"bytes://" +
                                  file_descriptor_set.SerializeToString())

        cls.SAMPLING_RATE_HZ = 16000.0

        # Sets up the wave header.
        wave_header = speech_data_pb2.WaveHeader()
        wave_header.number_of_channels = 1
        wave_header.sampling_rate_hz = cls.SAMPLING_RATE_HZ
        wave_header.atomic_type = speech_data_pb2.WaveHeader.INT16

        # yapf: disable
        cls.acoust_dict = {}
        cls.acoust_dict["SEQ_DATA"] = tf.constant(
            [[ 0,  1,  2,  3,  0],
             [ 5,  6,  7,  8,  0],
             [10, 11, 12, 13, 14],
             [15, 16, 17,  0,  0]], dtype=tf.dtypes.int16)
        # yapf: disable
        cls.acoust_dict["SEQ_LEN"] = tf.constant([4, 4, 5, 3])

        cls.label_dict = {}
        cls.label_dict["SEQ_DATA"] = tf.constant(["HELLO", "LOVE", "KIND", "HAPPY"])
        cls.label_dict["SEQ_LEN"] = tf.constant([1, 1, 1, 1])

        serialized_speech_data_list = []
        speech_data = speech_data_pb2.SpeechData()
        for i in range(tf.shape(cls.acoust_dict["SEQ_DATA"])[0]):
            speech_data.utterance_id = uuid.uuid4().hex
            speech_data.wave_header.CopyFrom(wave_header)
            length = cls.acoust_dict["SEQ_LEN"][i]

            speech_data.samples = cls.acoust_dict["SEQ_DATA"][i][:length].numpy().tobytes()
            speech_data.transcript = cls.label_dict["SEQ_DATA"][i].numpy()
            serialized_speech_data_list.append(speech_data.SerializeToString())

        cls._speech_data_str = tf.constant(serialized_speech_data_list)

    def test_speech_data_to_wave(self):
        op =speech_data_helper.SpeechDataToWave(tf.dtypes.float32)
        actual_output = op.process(self._speech_data_str)

        acoust_data = tf.cast(self.acoust_dict["SEQ_DATA"], dtype=tf.dtypes.float32) / 2 ** 15

        acoust_dict = {}
        acoust_dict["SEQ_LEN"] = self.acoust_dict["SEQ_LEN"]
        acoust_dict["SEQ_DATA"] = tf.expand_dims(acoust_data, axis=2)
        acoust_dict["SAMPLING_RATE_HZ"] = tf.fill((4), self.SAMPLING_RATE_HZ)

        label_dict = {}
        label_dict["SEQ_DATA"] = self.label_dict["SEQ_DATA"]
        label_dict["SEQ_LEN"] = tf.fill((4), 1)

        expected_output = (acoust_dict, label_dict)

        self.assertAllClose(expected_output[0], actual_output[0])
        self.assertAllEqual(expected_output[1]["SEQ_DATA"], actual_output[1]["SEQ_DATA"])
        self.assertAllEqual(expected_output[1]["SEQ_LEN"], actual_output[1]["SEQ_LEN"])


    def test_parse_speech_data(self):
        for i in range(4):
            actual_output = speech_data_helper.parse_speech_data(
                self._speech_data_str[i], self._string_descriptor,
                tf.dtypes.float32)
            acoust_data = tf.cast(
                self.acoust_dict["SEQ_DATA"], dtype=tf.dtypes.float32) / 2 ** 15

            acoust_dict = {}
            acoust_dict["SEQ_LEN"] = self.acoust_dict["SEQ_LEN"][i]
            acoust_dict["SEQ_DATA"] = tf.RaggedTensor.from_tensor(tf.reshape(
                acoust_data[i][:acoust_dict["SEQ_LEN"]], (-1, 1)))
            acoust_dict["SAMPLING_RATE_HZ"] = tf.constant(self.SAMPLING_RATE_HZ)

            label_dict = {}
            label_dict["SEQ_DATA"] = self.label_dict["SEQ_DATA"][i]
            label_dict["SEQ_LEN"] = tf.constant(1)

            expected_output = (acoust_dict, label_dict)

            self.assertAllClose(expected_output[0], actual_output[0])
            self.assertAllEqual(expected_output[1], actual_output[1])





if __name__ == "__main__":
    tf.test.main()
