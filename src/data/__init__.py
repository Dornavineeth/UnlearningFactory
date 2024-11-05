from typing import Dict, Any
from omegaconf import DictConfig

from data.tofu import (
    QADataset,
    QAwithIdkDataset,
)
from data.collators import (
    DataCollatorForSupervisedDataset,
)
from data.unlearn import ForgetRetainDataset


DATASET_REGISTRY: Dict[str, Any] = {}
COLLATOR_REGISTRY: Dict[str, Any] = {}


def _register_data(data_name, data_class):
    DATASET_REGISTRY[data_name] = data_class


def _register_collator(collator_name, collator_class):
    COLLATOR_REGISTRY[collator_name] = collator_class


def _load_single_dataset(dataset_name: str, data_cfg: DictConfig, **kwargs):
    dataset = DATASET_REGISTRY.get(dataset_name)
    if dataset is None:
        raise NotImplementedError(f"{dataset_name} not implemented or not registered")
    data_args = data_cfg.args
    return dataset(**data_args, **kwargs)


def get_datasets(dataset_cfgs: DictConfig, **kwargs):
    dataset = {}
    for dataset_name, data_cfg in dataset_cfgs.items():
        dataset[dataset_name] = _load_single_dataset(dataset_name, data_cfg, **kwargs)
    if len(dataset) == 1:
        # return a single dataset
        return list(dataset.values())[0]
    # return a multiple datasets in dictionary
    return dataset


def get_data(data_cfg: DictConfig, mode="train", **kwargs):
    data = {}
    for split, dataset_cfgs in data_cfg.items():
        data[split] = get_datasets(dataset_cfgs, **kwargs)
    if mode == "train":
        return data
    elif mode == "unlearn":
        unlearn_splits = {k: v for k, v in data.items() if k not in ("eval", "test")}
        unlearn_dataset = ForgetRetainDataset(**unlearn_splits)
        data["train"] = unlearn_dataset
        for split in unlearn_splits:
            data.pop(split)
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
        # return a single collator
        return list(collators.values())[0]
    # return collators in a dict
    return collators


# Register Datasets
_register_data("TOFU_QA_FULL", QADataset)
_register_data("TOFU_QA_FORGET10", QADataset)
_register_data("TOFU_QA_FORGET10_P", QADataset)
_register_data("TOFU_QA_FORGET10_PT", QADataset)
_register_data("TOFU_QA_FULL_PARAPHRASED10", QADataset)
_register_data("TOFU_QA_BIO", QADataset)
_register_data("TOFU_QA_RETAIN90", QADataset)
_register_data("TOFU_QAwithIdk_FORGET10", QAwithIdkDataset)

# Register Composite Datasets
# groups : unlearn
_register_data("TOFU_QA_FORGET10_RETAIN90", ForgetRetainDataset)

# Register Collators
_register_collator("DataCollatorForSupervisedDataset", DataCollatorForSupervisedDataset)
