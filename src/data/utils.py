import torch
import datasets
import numpy as np

IGNORE_INDEX = -100  # TODO put in common constants


def load_hf_dataset(path, **kwargs):
    dataset = datasets.load_dataset(path, **kwargs)
    return dataset


def package_prompt_response(
    template_config,
    tokenizer,
    prompt_msgs,
    response_msgs,
    max_length,
    predict_with_generate=False,
):
    """prompt_msgs and response_msgs are lists where except the last pair, all 
    corresponding pairs are in-context examples. When they are a string and not
    a list, there are no in-context examples."""
    assert len(prompt_msgs) == len(response_msgs)
    if isinstance(prompt_msgs, str):
        assert isinstance(response_msgs, str)
        prompt_msgs, response_msgs = [prompt_msgs], [response_msgs]
    
    if template_config["apply_chat_template"]:
        chat = []
        system_prompt = template_config.get("system_prompt", None)
        if system_prompt:
            chat += [{"role": "system", "content": system_prompt}]
        for prompt, response in zip(prompt_msgs, response_msgs):
            chat += [{"role": "user", "content": prompt}]
            chat += [{"role": "assistant", "content": response}]
        chat_ids = tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=False
        )
        # all except last response are in-context examples
        wrapped_prompt = tokenizer.apply_chat_template(
            chat[:-1], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.apply_chat_template(
            chat[:-1], tokenize=True, add_generation_prompt=True
        )
    else:
        wrapped_prompt = ""
        
        # add in-context examples
        n_few_shot = len(prompt_msgs)-1
        for i in range(n_few_shot):
            fs_prompt, fs_response = prompt_msgs[i], response_msgs[i]
            wrapped_prompt += (
                template_config["user_start_tag"]
                + fs_prompt
                + template_config["user_end_tag"]
                + template_config["asst_tag"]
                + fs_response
                + template_config["example_separator"]
            )
        
        # add actual example
        final_prompt, final_response = prompt_msgs[-1], response_msgs[-1]
        wrapped_prompt += (
            template_config["user_start_tag"]
            + final_prompt
            + template_config["user_end_tag"]
            + template_config["asst_tag"]
        )
        chat_ids = tokenizer(
            wrapped_prompt + final_response,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
        )["input_ids"]
        
        prompt_ids = tokenizer(
            wrapped_prompt,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
        )["input_ids"]

    if chat_ids[-1] != tokenizer.eos_token_id:
        chat_ids += [tokenizer.eos_token_id]

    if template_config["asst_tag"] != "":  ## for llama2-chat model don't assert
        assert chat_ids[: len(prompt_ids)] == prompt_ids, ValueError(
            "Tokenization mismatch: tokenized prompt should be a prefix of tokenized prompt+response. Discrepancy usually arises around the last prompt index."
        )

    labels = [IGNORE_INDEX] * len(prompt_ids) + chat_ids[len(prompt_ids) :]
    item = {}
    if predict_with_generate:
        item["input_ids"] = prompt_ids
    else:
        item["input_ids"] = chat_ids
    item["labels"] = labels
    item["attention_mask"] = [1] * len(item["input_ids"])
    for attr in item:
        item[attr] = torch.tensor(item[attr])
    return item


def add_dataset_index(dataset):
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column("index", indexing)
    return dataset
