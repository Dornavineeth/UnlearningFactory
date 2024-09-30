
from dataclasses import dataclass
from typing import Dict, Sequence
import transformers
import torch

IGNORE_INDEX = -100 # TODO put in constants

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(input_ids=input_ids, labels=labels, 
                    attention_mask=input_ids.ne(self.tokenizer.pad_token_id))

@dataclass
class DataCollatorForSupervisedDatasetWithIndices(DataCollatorForSupervisedDataset):
    """Collate examples for supervised fine-tuning and handle example indices."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        collated = super().__call__(instances)
        collated['indices'] = torch.stack([example['indices'] for example in instances])
        return collated
