from typing import Dict, Any
from omegaconf import DictConfig

from data.tofu import QADataset
from data.collators import (
    DataCollatorForSupervisedDataset,
    DataCollatorForSupervisedDatasetWithIndex,
)


DATA_REGISTRY: Dict[str, Any] = {}
COLLATOR_REGISTRY: Dict[str, Any] = {}


def _register_data(data_name, data_class):
    DATA_REGISTRY[data_name] = data_class


def _register_collator(collator_name, collator_class):
    COLLATOR_REGISTRY[collator_name] = collator_class


# TODO add tofu wiki support
def _load_single_dataset(dataset_name: str, data_cfg: DictConfig, **kwargs):
    dataset = DATA_REGISTRY.get(dataset_name)
    if dataset is None:
        raise NotImplementedError(f"{dataset_name} not implemented or not registered")
    data_args = data_cfg.args
    return dataset(**data_args, **kwargs)


def get_datasets(data_cfgs: DictConfig, **kwargs):
    data = {}
    for dataset_name, data_cfg in data_cfgs.items():
        data[dataset_name] = _load_single_dataset(dataset_name, data_cfg, **kwargs)
    if len(data) == 1:
        # return a single dataset
        return list(data.values())[0]
    # return a multiple datasets in dictionary
    return data


def _get_single_collator(collator_name: str, collator_cfg: DictConfig, **kwargs):
    collator = COLLATOR_REGISTRY.get(collator_name)
    if collator is None:
        raise NotImplementedError(f"{collator_name} not implemented or not registered")
    collator_args = collator_cfg.args
    return collator(**collator_args, **kwargs)


def get_collators(collator_cfgs, **kwargs):
    collators = {}
    for collator_name, collator_cfg in collator_cfgs.items():
        collators[collator_name] = _get_single_collator(
            collator_name, collator_cfg, **kwargs
        )
    if len(collators) == 1:
        # return a single dataset
        return list(collators.values())[0]
    # return a multiple datasets in dictionary
    return collators


# Register Datasets
_register_data("TOFU_QA_FULL", QADataset)
_register_data("TOFU_QA_FORGET10", QADataset)
_register_data("TOFU_QA_FORGET10_P", QADataset)
_register_data("TOFU_QA_FORGET10_PT", QADataset)
_register_data("TOFU_QA_FULL_PARAPHRASED10", QADataset)
_register_data("TOFU_QA_BIO", QADataset)

# Register Collators
_register_collator("DataCollatorForSupervisedDataset", DataCollatorForSupervisedDataset)
_register_collator(
    "DataCollatorForSupervisedDatasetWithIndex",
    DataCollatorForSupervisedDatasetWithIndex,
)
