from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model(model_cfg: Dict):
    model_name = model_cfg.get("name", None)
    model_args = model_cfg.get("args", None)
    if model_name in ["llama3_1_instruct"]:
        model = AutoModelForCausalLM.from_pretrained(**model_args)
    else:
        raise NotImplementedError(
            f"{model_name} is not supported or Please use a valid model_name"
        )
    return model

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
    tokenizer_name = tokenizer_cfg.get("name", None)
    tokenizer_args = tokenizer_cfg.get("args", None)
    if tokenizer_name in ["llama3_1_instruct"]:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args)
    else:
        raise NotImplementedError(
            f"{tokenizer_name} is not supported or Please use a valid tokenizer_name"
        )
    if tokenizer.eos_token_id is None:
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Add pad token: {}".format(tokenizer.pad_token))
    return tokenizer
