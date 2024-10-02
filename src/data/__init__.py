from omegaconf import DictConfig

from .collators import (
    DataCollatorForSupervisedDataset,
    DataCollatorForSupervisedDatasetWithIndex,
)
from .tofu import TOFU_QA

def _load_single_dataset(dataset_name: str,data_cfg: DictConfig, **kwargs):
    data_args = data_cfg.args
    if dataset_name in ["TOFU_QA_FULL", "TOFU_QA_FORGET10", "TOFU_QA_FORGET10_P"]:
        return TOFU_QA(**data_args, **kwargs)
    else:
        raise NotImplementedError(dataset_name)

# TODO add tofu wiki support
def get_datasets(data_cfgs: DictConfig, **kwargs):
    data = {}
    for dataset_name, data_cfg in data_cfgs.items():
        data[dataset_name] = _load_single_dataset(dataset_name, data_cfg, **kwargs)
    if len(data)==1:
        # return a single dataset
        return list(data.values())[0]
    # return a multiple datasets in dictionary
    return data

def _get_single_collator(collator_name: str, collator_cfg: DictConfig, **kwargs):
    collator_args = collator_cfg.args
    if collator_name=="DataCollatorForSupervisedDataset":
        return DataCollatorForSupervisedDataset(**collator_args, **kwargs)
    elif collator_name=="DataCollatorForSupervisedDatasetWithIndex":
        return DataCollatorForSupervisedDatasetWithIndex(**collator_args, **kwargs)
    else:
        raise NotImplementedError
    
def get_collators(collator_cfgs, **kwargs):
    collators = {}
    for collator_name, collator_cfg in collator_cfgs.items():
        collators[collator_name] = _get_single_collator(collator_name, collator_cfg, **kwargs)
    if len(collators)==1:
        # return a single dataset
        return list(collators.values())[0]
    # return a multiple datasets in dictionary
    return collators