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
    import tensorflow as tf
    import tensorflow_text as tf_text
except ImportError as e:
    print("There are some packages need to be installed.")
from packaging import version

# Custom imports
from data.operation import text_codec_params
from operation import operation

assert version.parse(tf.__version__) >= version.parse("2.0.0"), (
    "at least tensorflow 2.0 is required.")


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

        serialized_model = open(self._params.model_name, "rb").read()

        self._sp_tokenizer = tf_text.SentencepieceTokenizer(
            model=serialized_model,
            out_type=tf.int32,
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

    def process(self, inputs):
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

        assert tf.is_tensor(inputs["SEQ_DATA"]), (
            "inputs[\"SEQ_DATA\"] must be a Tensor type.")
        assert tf.is_tensor(
            inputs["SEQ_LEN"]), ("inputs[\"SEQ_LEN\"] must be a Tensor type.")

        outputs = {}
        if self._params.processing_mode == (
                text_codec_params.ProcessingMode.ENCODING):
            data = self._sp_tokenizer.tokenize(inputs["SEQ_DATA"])
            outputs["SEQ_LEN"] = data.row_lengths()
            outputs["SEQ_DATA"] = data.to_tensor()
        elif self._params.processing_mode == (
                text_codec_params.ProcessingMode.DECODING):
            # Converts the input into a RaggedTensor.
            if not isinstance(inputs["SEQ_DATA"], tf.RaggedTensor):
                inputs["SEQ_DATA"] = tf.RaggedTensor.from_tensor(
                    inputs["SEQ_DATA"], lengths=inputs["SEQ_LEN"])

            data = self._sp_tokenizer.detokenize(inputs["SEQ_DATA"])
            outputs["SEQ_DATA"] = data
            outputs["SEQ_LEN"] = tf.ones((inputs["SEQ_DATA"].nrows(), ),
                                         dtype=tf.dtypes.int32)

        return outputs
