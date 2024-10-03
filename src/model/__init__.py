from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig
import os
import torch
hf_home = os.getenv('HF_HOME', default=None)


def get_dtype(trainer_args=None, dtype_str=None):
    if trainer_args is None and dtype_str is None:
        return torch.float32
    elif trainer_args is None:
        if dtype_str == 'bf16':
            return torch.bfloat16
        elif dtype_str == 'fp16':
            return torch.float16
        else:
            return torch.float32
    elif trainer_args.bf16:
        return torch.bfloat16
    elif trainer_args.fp16:
        return torch.float16
    else:
        return torch.float32


def get_model(model_cfg: DictConfig, torch_dtype):
    assert model_cfg is not None and model_cfg.model_args is not None, ValueError("Model config not found or model_args absent in configs/model.")
    model_args = model_cfg.model_args
    tokenizer_args = model_cfg.tokenizer_args
    try:
        model = AutoModelForCausalLM.from_pretrained(**model_args, torch_dtype=torch_dtype, cache_dir=hf_home)
        print(f"Fetched {model_args.pretrained_model_name_or_path}")
    except Exception as e:
        print(f"Model {model_args.pretrained_model_name_or_path} requested with")
        print(model_cfg.model_args)
        print(f"Error {e} while fetching model using AutoModelForCausalLM.from_pretrained().")
        raise
    tokenizer = get_tokenizer(tokenizer_args)
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


def get_tokenizer(tokenizer_cfg: DictConfig):
    try:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_cfg, cache_dir=hf_home)
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
