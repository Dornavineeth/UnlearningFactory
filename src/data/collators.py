
from dataclasses import dataclass
from typing import Dict, Sequence
import transformers
import torch

IGNORE_INDEX = -100 # TODO put in constants

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, padding_side: str = 'right'):
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        
    def _pad_tokens(self, input_ids, padding_value):
        if self.padding_side == 'right':
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=padding_value
            )
        else:
            input_ids = torch.nn.utils.rnn.pad_sequence([
                torch.flip(i, dims=0) for i in input_ids 
            ], batch_first=True, padding_value=padding_value).flip(dims=[1])                    
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return_dct = {}
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        assert len(instances)>=0
        if 'input_ids' in instances[0]:
            input_ids = [instance['input_ids'] for instance in instances] 
            input_ids = self._pad_tokens(input_ids, self.tokenizer.pad_token_id)
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            return_dct.update({"input_ids":input_ids})
            return_dct.update({"attention_mask":attention_mask})
        if 'labels' in instances[0]:
            labels = [instance['labels'] for instance in instances] 
            labels = self._pad_tokens(labels, IGNORE_INDEX)
            return_dct.update({"labels":labels})
        
        return return_dct

class DataCollatorForSupervisedDatasetWithIndex(DataCollatorForSupervisedDataset):
    """Collate examples for supervised fine-tuning and handle example indices."""

    tokenizer: transformers.PreTrainedTokenizer
    padding_side: str = 'left'
    index: str = 'index'
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        collated = super().__call__(instances)
        collated[self.index] = torch.stack([example[self.index] for example in instances])
        return collated
