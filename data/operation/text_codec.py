"""A module defining text codecs.

TODO(chanwcom)
Adds more explanation.
"""
# pylint: disable=import-error, no-member

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import os

# Third-party imports
try:
    import sentencepiece as sp
    import torch
except ImportError as e:
    print("There are some packages need to be installed.")
from packaging import version

# Custom imports
from data.operation import text_codec_params
from operation import operation


def make_padded_batch(inputs: [torch.Tensor],
                      padding_value: int = 0) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)


def padded_batch_to_ragged_list(inputs: list, padding_value: int = 0) -> list:
    # Padding value must be smaller than all other values.
    lengths = [torch.sum(torch.tensor(row) > padding_value) for row in inputs]
    return [row[0:row_length] for row_length, row in zip(lengths, inputs)]


# TODO(chanw.com) This class should be derived from a TextCodec class
# This should be an abstract class .
class SentencePieceTextCodec(operation.AbstractOperation):
    """An operation class for encoding and decoding using SentencePiece."""

    def __init__(self, params: text_codec_params.TextCodecParams):
        """Initializes a SentencePieceTextCodec object.

            Args:
                params: A proto-message for initializing the object.

            Returns:
                None.
        """
        assert isinstance(params, text_codec_params.TextCodecParams)
        self._params = params

        if not os.path.isfile(self._params.model_name):
            raise FileExistsError("{0} does not exist.".format(
                self._params.model_name))

        self._sp_tokenizer = sp.SentencePieceProcessor(
            model_file=self._params.model_name,
            out_type=int,
            add_bos=self._params.add_bos,
            add_eos=self._params.add_eos)

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        # Unpacks the any proto message.
        assert isinstance(params, text_codec_params.TextCodecParams)

        self._params = params

    def process(self, inputs: dict):
        """Returns the output tensor given an input tensor.

        The operation may be either encoding or decoding.

        Args:
            inputs: A dictionary containing data with the "SEQ_DATA" key.
                The shape of inputs["SEQ_DATA"] is (batch_size,
                sequence_length). For tokenization, sequence_length is usually
                one. For detokenization, sequence length is usually larger than
                one.

        Returns:
            A dictionary containing "SEQ_DATA" and "SEQ_LEN".
                SEQ_DATA: The output of either tokenization or
                    detokenization.  The shape is (batch_size, max_length).
                    Each example corresponds to the row of this tensor, and may
                    be zero-padded.
                SEQ_LEN: The length of each example in a batch.
                    The shape is (batch_size,)
        """
        assert (isinstance(inputs, dict) and {
            "SEQ_DATA", "SEQ_LEN"
        } <= inputs.keys()), (
            "The inputs to this method must be a dictionary having \"SEQ_DATA\" "
            "and \"SEQ_LEN\" as keys.")

        if isinstance(inputs["SEQ_DATA"], torch.Tensor):
            inputs_data = inputs["SEQ_DATA"].numpy().tolist()
        elif isinstance(inputs["SEQ_DATA"], list):
            inputs_data = inputs["SEQ_DATA"]
        else:
            raise ValueError("Unsupported type!")

        outputs = {}
        if self._params.processing_mode == (
                text_codec_params.ProcessingMode.ENCODING):
            data = self._sp_tokenizer.tokenize(inputs_data)
            outputs["SEQ_LEN"] = torch.tensor([len(row) for row in data])
            outputs["SEQ_DATA"] = make_padded_batch(
                [torch.tensor(row) for row in data])
        elif self._params.processing_mode == (
                text_codec_params.ProcessingMode.DECODING):
            outputs["SEQ_DATA"] = self._sp_tokenizer.decode(
                padded_batch_to_ragged_list(inputs_data))
            num_rows = len(inputs_data)
            outputs["SEQ_LEN"] = torch.ones((num_rows, ), dtype=torch.int32)

        return outputs
