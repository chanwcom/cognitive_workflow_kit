"""A module for unit-testing the "speech_data_helper" module."""

# pylint: disable=invalid-name, no-member, import-error
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import os

# Third-party imports
import tensorflow as tf
import soundfile
from packaging import version

# Custom imports
from codelab.unittest_template import example_tf

assert version.parse(tf.__version__) >= version.parse("2.0.0"), (
    "At least tensorflow 2.0 is required.")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class WaveToSpeechDataTest(tf.test.TestCase):
    """A class for testing methods in the example module."""

    @classmethod
    def setUpClass(cls):
        cls._wave_data_list = []

        wave_header = speech_data.WaveHeader()
        wave_header.number_of_channels = 2
        wave_header.sampling_rate_hz = 16000.0
        wave_header.atomic_type = speech_data.FLOAT32
        data = np.numpy([[2.0, 1.0], [3.0, 4.0], [5.0, 3.0], [1.0, 0.0]])
        transcript = "HELLO"

        cls._wave_data_list.append((wave_header, data, transcript))

        wave_header.number_of_channels = 1
        wave_header.sampling_rate_hz = 16000.0
        wave_header.atomic_type = speech_data.INT16
        data = np.numpy([[2], [3], [5], [1]])
        transcript = "MY"

        cls._wave_data_list.append((wave_header, data, transcript))


    def test_two_channel_audio(self):
        """This method tests the "resize_tensor" method in a simple case."""
        op = speech_data_helper.WaveToSpeechData()

        actual_list = op.process(self._wave_data_list)

        # Checks whether the length is two.
        self.assertEqual(self._wave_data_list[0][0], actual[0].wave_header)
        self.assertTrue(istype(actual[0].utterance_id, str))
        self.assertEqual(_wave_data_list[0][1],
                         np.reshape(np.frombuffer(actual[0].samples, dtype=float), (-1, 2)))
        self.assertEqual("HELLO", actual[0].transcript)

        self.assertEqual(self._wave_data_list[1][0], actual[1].wave_header)
        self.assertTrue(istype(actual[1].utterance_id, str))
        self.assertEqual(_wave_data_list[1][1],
                         np.reshape(np.frombuffer(actual[1].samples, dtype=int), (-1, 1)))
        self.assertEqual("MY", actual[1].transcript)


if __name__ == "__main__":
    tf.test.main()
