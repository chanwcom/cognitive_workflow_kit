#!/usr/bin/python3
"""A module for unit-testing the text_codec module."""

# pylint: disable=invalid-name, no-member, import-error
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import os

# Third-party imports
import torch
import unittest
from packaging import version

# Custom imports
from data.operation import text_codec
from data.operation import text_codec_params

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def module_dir():
    """Returns the full directory name of the current model."""
    return os.path.dirname(os.path.abspath(__file__))


class SentencePieceTextCodecTest(unittest.TestCase):
    """A class for unit-testing the SentencePieceTextCodec class."""

    @classmethod
    def setUpClass(cls):
        cls._model_file_name = os.path.join(
            module_dir(), "testdata/model_unigram_256.model")

    def test_encode(self):
        params = text_codec_params.TextCodecParams(
            self._model_file_name, text_codec_params.ProcessingMode.ENCODING)

        op_obj = text_codec.SentencePieceTextCodec(params)

        inputs = {}
        inputs["SEQ_DATA"] = [u"HELLO BIXBY", u"VIDEO MUSIC", u"GOOD MUSIC"]
        inputs["SEQ_LEN"] = torch.tensor([1, 1, 1], dtype=torch.int32)

        actual_output = op_obj.process(inputs)

        expected_output = {}
        # yapf: disable
        expected_output["SEQ_DATA"] = torch.tensor(
            [[1,  29, 59, 11, 50,  9, 200, 38, 19,  2,   0, 0],
             [1,  15, 54,  9,  7,  4,  11, 15, 14, 96, 105, 2],
             [1, 245, 15, 14, 96, 105,  2,  0,  0,  0,   0, 0]],
            dtype=torch.int32)
        # yapf: enable
        expected_output["SEQ_LEN"] = torch.tensor([10, 12, 7])

        self.assertEqual(expected_output.keys(), actual_output.keys())
        self.assertTrue(
            torch.equal(expected_output["SEQ_DATA"],
                        actual_output["SEQ_DATA"]))
        self.assertTrue(
            torch.equal(expected_output["SEQ_LEN"], actual_output["SEQ_LEN"]))

    def test_decode_ragged_tensor(self):
        """Tests the decoding procedure when the input is a ragged tensor."""
        params = text_codec_params.TextCodecParams(
            self._model_file_name, text_codec_params.ProcessingMode.DECODING)

        op_obj = text_codec.SentencePieceTextCodec(params)

        inputs = {}

        # The following inputs is the "expected_output" of the above
        # "text_encode" unit test.
        # yapf: disable
        inputs["SEQ_DATA"] = [
            [1,  29, 59, 11, 50,   9, 200, 38, 19,  2],
            [1,  15, 54,  9,  7,   4,  11, 15, 14, 96, 105, 2],
            [1, 245, 15, 14, 96, 105,   2]]
        # yapf: enable
        inputs["SEQ_LEN"] = [len(row) for row in inputs["SEQ_DATA"]]
        actual_output = op_obj.process(inputs)

        expected_output = {}
        expected_output["SEQ_DATA"] = [
            u"HELLO BIXBY", u"VIDEO MUSIC", u"GOOD MUSIC"
        ]
        expected_output["SEQ_LEN"] = torch.tensor([1, 1, 1])

        self.assertEqual(expected_output.keys(), actual_output.keys())
        self.assertListEqual(expected_output["SEQ_DATA"],
                             actual_output["SEQ_DATA"])
        self.assertTrue(
            torch.equal(expected_output["SEQ_LEN"], actual_output["SEQ_LEN"]))

    def test_decode_normal_tensor(self):
        """Tests the decoding procedure when the input is a normal tensor."""
        params = text_codec_params.TextCodecParams(
            self._model_file_name, text_codec_params.ProcessingMode.DECODING)

        op_obj = text_codec.SentencePieceTextCodec(params)

        inputs = {}

        # The following inputs is the "expected_output" of the above
        # "text_encode" unit test.
        # yapf: disable
        inputs["SEQ_DATA"] = torch.tensor(
            [[1,  29, 59, 11, 50,   9, 200, 38, 19,  2,   0, 0],
             [1,  15, 54,  9,  7,   4,  11, 15, 14, 96, 105, 2],
             [1, 245, 15, 14, 96, 105,   2,  0,  0,  0,   0, 0]],
            dtype=torch.int32)
        # yapf: enable
        inputs["SEQ_LEN"] = torch.tensor([10, 12, 7], dtype=torch.int32)
        actual_output = op_obj.process(inputs)

        expected_output = {}
        expected_output["SEQ_DATA"] = [
            u"HELLO BIXBY", u"VIDEO MUSIC", u"GOOD MUSIC"
        ]
        expected_output["SEQ_LEN"] = torch.tensor([1, 1, 1], dtype=torch.int32)

        self.assertEqual(expected_output.keys(), actual_output.keys())
        self.assertListEqual(expected_output["SEQ_DATA"],
                             actual_output["SEQ_DATA"])
        self.assertTrue(
            torch.equal(expected_output["SEQ_LEN"], actual_output["SEQ_LEN"]))


if __name__ == "__main__":
    unittest.main()
