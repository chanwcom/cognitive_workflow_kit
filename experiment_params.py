__author__ = "Chanwoo Kim(chanwcom@gmail.com)"
# Standard imports
from dataclasses import dataclass
from enum import Enum

# Custom imports
from operation import operation_params

# Let's use
# https://pypi.org/project/dataclasses-json/
# https://pypi.org/project/dataclasses-json/


@dataclass
class ExperimentParams(operation_params.AbstractOpParams):
    experiment_name
    backbone_type: = P ,
    working_dir: str,
    trainer_params: TrainerParams,
    tester_params: TesterParams,
    distributed_processing_
