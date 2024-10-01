import numpy as np
import torch

IGNORE_INDEX = -100 # TODO put in common constants

def package_prompt_response(template_config, tokenizer, prompt, response, max_length):
    if template_config["apply_chat_template"]:
        chat = []
        system_prompt = template_config.get("system_prompt", None)
        if system_prompt:
            chat += [{"role": "system", "content": system_prompt}]
        chat += [{"role": "user", "content": prompt}, 
                {"role": "assistant", "content": response}]
        chat_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False)
        wrapped_prompt = tokenizer.apply_chat_template(chat[:-1], tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.apply_chat_template(chat[:-1], tokenize=True, add_generation_prompt=True)
    else:
        wrapped_prompt = template_config["user_start_tag"] + prompt + template_config["user_end_tag"] + template_config["asst_tag"]
        chat_ids = tokenizer(wrapped_prompt + response, add_special_tokens=True, max_length=max_length, truncation=True)['input_ids']
        prompt_ids = tokenizer(wrapped_prompt, add_special_tokens=True, max_length=max_length, truncation=True)['input_ids']
    
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

