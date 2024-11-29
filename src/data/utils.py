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
    prompts,
    responses,
    max_length,
    predict_with_generate=False,
):
    # when there are multiple prompts and responses, except the last pair, all 
    # corresponding pairs are in-context examples
    assert len(prompts) == len(responses)
    if isinstance(prompts, str):
        assert isinstance(responses, str)
        prompts, responses = [prompts], [responses]
    
    if template_config["apply_chat_template"]:
        chat = []
        system_prompt = template_config.get("system_prompt", None)
        if system_prompt:
            chat += [{"role": "system", "content": system_prompt}]
        for prompt, response in zip(prompts, responses):
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
        n_few_shot = len(prompts)-1
        wrapped_prompt = ""
        for i in range(n_few_shot):
            fs_prompt, fs_response = prompts[i], responses[i]
            wrapped_prompt += (
                template_config["user_start_tag"]
                + fs_prompt
                + template_config["user_end_tag"]
                + template_config["asst_tag"]
                + fs_response
                + template_config["example_separator"]
            )
        final_prompt, final_response = prompts[-1], responses[-1]
        
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
