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
    add_bos: bool
    add_eos: bool
    processing_mode: ProcessingMode
