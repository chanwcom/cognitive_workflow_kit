"""A module for unit-testing the "speech_data_helper" module."""

# pylint: disable=invalid-name, no-member, import-error
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import unittest

# Third-party imports
import numpy as np

# Custom imports
from data.format import speech_data_helper
from data.format import speech_data_pb2


class WaveToSpeechDataTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
