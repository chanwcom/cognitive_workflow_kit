__author__ = "Chanwoo Kim(chanwcom@gmail.com)"
# Standard imports
from dataclasses import dataclass
from enum import Enum

# Custom imports
from operation import operation_params


class ProcessingMode(Enum):
    ENCODING = 1
    DECODING = 2


@dataclass
class TextCodecCreationParams(operation_params.AbstractOpCreationParams):
    class_name: str = "TextCodec"
    class_params: TextCodecParams


@dataclass
class TextCodecParams(operation_params.AbstractOpParams):
    model_name: str
    processing_mode: ProcessingMode
    add_bos: bool = True
    add_eos: bool = True
    padding_value: int = 0
