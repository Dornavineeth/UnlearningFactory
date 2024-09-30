from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict
import os
import yaml

# TODO: any better way to do this? Couldn't figure out.
def get_config_from_model_name(name) -> Dict:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(current_dir, f'../../configs/model/{name}.yaml')
    with open(config_file_path, 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    return config_data


def get_model(model_name):
    try:
        model_cfg = get_config_from_model_name(model_name)
        assert model_cfg is not None and 'model_args' in model_cfg
    except Exception as e:
        print(e)
        raise ValueError(f"Model config not found or model_args absent in configs/model for {model_name}.")
    try:
        model = AutoModelForCausalLM.from_pretrained(**model_cfg["model_args"])
    except Exception as e:
        print(f"Model requested with {model_cfg['pretrained_model_name_or_path']}")
        print(model_cfg["model_args"])
        print(f"Error {e} while fetching model using AutoModelForCausalLM.from_pretrained().")
        raise
    tokenizer = get_tokenizer(model_cfg["tokenizer_args"])
    return model, tokenizer


def _add_or_replace_eos_token(tokenizer, eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        print("Add eos token: {}".format(tokenizer.eos_token))
    else:
        print("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        print("New tokens have been added, make sure `resize_vocab` is True.")


def get_tokenizer(tokenizer_cfg: Dict):
    try:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_cfg)
    except Exception as e:
        print(f"Tokenizer requested with {tokenizer_cfg.pretrained_model_name_or_path}")
        print(tokenizer_cfg.model_args)
        print(f"Error {e} fetching tokenizer using AutoTokenizer.")
        raise
    
    if tokenizer.eos_token_id is None:
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Setting pad_token as eos token: {}".format(tokenizer.pad_token))
    
    return tokenizer
