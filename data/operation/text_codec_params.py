__author__ = "Chanwoo Kim(chanwcom@gmail.com)"
# Standard imports
from dataclasses import dataclass
from enum import Enum


class ProcessingMode(Enum):
    ENCODING = 1
    DECODING = 2


@dataclass
class TextCodecParams:
    model_name: str
    processing_mode: ProcessingMode
    add_bos: bool = True
    add_eos: bool = True
    padding_value: int = 0
