from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig


def get_model(model_cfg: DictConfig):
    try:
        model = AutoModelForCausalLM.from_pretrained(**model_cfg.model_args)
    except Exception as e:
        raise ValueError(f"Error {e} while fetching {model_cfg.pretrained_model_name_or_path} \
            using AutoModelForCausalLM.from_pretrained().")
    tokenizer = get_tokenizer(model_cfg.tokenizer_args)
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
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_cfg)
    except Exception as e:
        print(f"Error {e} fetching tokenizer with config", tokenizer_cfg)
        raise ValueError(f"{tokenizer_cfg.pretrained_model_name_or_path} \
            could be an invalid tokenizer path for AutoTokenizer.")
    
    if tokenizer.eos_token_id is None:
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Setting pad_token as eos token: {}".format(tokenizer.pad_token))
    return tokenizer
