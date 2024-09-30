from .tofu import TOFU_QA
from .collators import DataCollatorForSupervisedDataset, DataCollatorForSupervisedDatasetWithIndices

# TODO add tofu wiki support
def get_dataset(dataset_name, dataset_config, tokenizer_details):
    if dataset_name == "TOFU_QA_FULL":
        return TOFU_QA(**dataset_config, **tokenizer_details)
    else:
        raise NotImplementedError

# TODO: must take collator config instead
def get_collator(collator_name):
    if collator_name=="DataCollatorForSupervisedDataset":
        return DataCollatorForSupervisedDataset
    elif collator_name=="DataCollatorForSupervisedDatasetWithIndices":
        return DataCollatorForSupervisedDatasetWithIndices
    else:
        raise NotImplementedError