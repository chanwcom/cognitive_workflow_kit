#
#
__author__ = "Chanwoo Kim(chanwcom@gmail.com)"
# Standard imports
from dataclasses import dataclass
from enum import Enum

# Custom imports
from operation import operation_params


@dataclass
class DatasetOpCreationParams(operation_params.AbstractOpCreationParams):
    class_name: str
    class_params: operation_params.AbstractOpParams


@dataclass
class DatasetOpParams(operation_params.AbstractOpParams):
    pass


@dataclass
class NTPDatasetOpParams(DatasetOpParams):
    pass


@dataclass
class BatchDatasetOpParams(DatasetOpParams):
    batch_size: int = 1
    drop_remainder: bool = False


@dataclass
class PaddedBatchDatasetOpParams(DatasetOpParams):
    batch_size: int = 1
    drop_remainder: bool = False


@dataclass
class OptionalDatasetOpParams(DatasetOpParams):
    use_cache: bool = False
    use_prefetch: bool = False
