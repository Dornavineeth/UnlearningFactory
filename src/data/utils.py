import numpy as np
import torch
import yaml
from dataclasses import dataclass
from typing import Dict, Sequence
import transformers

IGNORE_INDEX = -100 # TODO put in constants

def get_model_cfg(model_family):
    model_configs  = {}
    with open("../../configs/model_configs.yaml", "r") as f: # TODO this will be useful in train code as well
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    return model_configs[model_family]

def package_prompt_response(model_config, tokenizer, prompt, response, max_length):
    try:
        if 'chat_templating' in model_config:
            chat = [{"role": "user", "content": prompt}, 
                    {"role": "assistant", "content": response}]
            if model_config['chat_templating']['system_role']:
                chat = [{"role": "system", "content": "You are a helpful assistant."}]+chat
            chat_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False)
            wrapped_prompt = tokenizer.apply_chat_template(chat[:-1], tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer.apply_chat_template(chat[:-1], tokenize=True, add_generation_prompt=True)
        else:
            wrapped_prompt = model_config['user_start_tag'] + prompt + model_config['user_end_tag'] + model_config['asst_tag']
            chat_ids = tokenizer(wrapped_prompt + response, add_special_tokens=True, max_length=max_length, truncation=True)['input_ids']
            prompt_ids = tokenizer(wrapped_prompt, add_special_tokens=True, max_length=max_length, truncation=True)['input_ids']
    except Exception as _:
        import pdb
        pdb.set_trace()
    
    if chat_ids[-1] != tokenizer.eos_token_id:
        chat_ids += [tokenizer.eos_token_id]
    
    assert chat_ids[:len(prompt_ids)] == prompt_ids
    labels = [IGNORE_INDEX]*len(prompt_ids) + chat_ids[len(prompt_ids):]
    item = {'input_ids': chat_ids, 'labels': labels}
    item['attention_mask'] = [1] * len(item['input_ids'])
    for attr in item:
        item[attr] = torch.tensor(item[attr])
    return item

def add_dataset_index(dataset):
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column("index", indexing)
    return dataset

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
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

@dataclass
class DataCollatorForSupervisedDatasetWithIndices(DataCollatorForSupervisedDataset):
    """Collate examples for supervised fine-tuning and handle example indices."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        collated = super().__call__(instances)
        collated['indices'] = torch.stack([example['indices'] for example in instances])
        return collated
