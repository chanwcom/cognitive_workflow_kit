__author__ = "Chanwoo Kim(chanwcom@gmail.com)"
# Standard imports
from dataclasses import dataclass
import enum

# Custom imports
from operation import operation_params

# TODO(chanwcom)
# Let's use the following approach
# https://pypi.org/project/dataclasses-json/
# https://stackoverflow.com/questions/51286748/make-the-python-json-encoder-support-pythons-new-dataclasses


class BackboneType(enum.Enum):
    """A class defining the backbone Machine Learning (ML) tool kit type."""
    TF2_KERAS = 1
    PYTORCH_KERAS = 2


@dataclass
class ExperimentParams(operation_params.AbstractOpParams):
    experiment_name: str
    backbone_type: BackboneType = BackboneType.TF2_KERAS
    working_dir: str
    model_params: ModelParams
    model_checkpoint_params: ModelCheckpointParams
    dataset_op_dict_params: DatasetOpDictParams
    trainer_params: TrainerParams
    tester_params: TesterParams
