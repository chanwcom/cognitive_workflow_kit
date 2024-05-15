"""A model implementing helpder methods for the SpeechData proto-message."""

# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import uuid

# Third party imports
import tensorflow as tf
from packaging import version

# Custom imports
import operation

assert version.parse(tf.__version__) >= version.parse("2.0.0"), (
    "At least tensorflow 2.0 is required.")


class WaveToSpeechData(operation.Operation):
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
