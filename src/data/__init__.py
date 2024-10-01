from omegaconf import DictConfig

from .collators import (
    DataCollatorForSupervisedDataset,
    DataCollatorForSupervisedDatasetWithIndices,
)
from .tofu import TOFU_QA


# TODO add tofu wiki support
def get_dataset(data_cfg: DictConfig, **kwargs):
    dataset_name = data_cfg.name
    data_args = data_cfg.data_args
    if dataset_name == "TOFU_QA_FULL":
        return TOFU_QA(**data_args, **kwargs)
    else:
        raise NotImplementedError

# TODO: must take collator config instead
def get_collator(collator_cfg, **kwargs):
    name = collator_cfg.name
    collator_args = collator_cfg.get("collator_args", {})
    if name=="DataCollatorForSupervisedDataset":
        return DataCollatorForSupervisedDataset(**collator_args, **kwargs)
    elif name=="DataCollatorForSupervisedDatasetWithIndices":
        return DataCollatorForSupervisedDatasetWithIndices(**collator_args, **kwargs)
    else:
        raise NotImplementedError