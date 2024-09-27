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


def get_tokenizer(tokenizer_cfg: Dict):
    tokenizer_name = tokenizer_cfg.get("name", None)
    tokenizer_args = tokenizer_cfg.get("args", None)
    if tokenizer_name in ["llama3_1_instruct"]:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args)
    else:
        raise NotImplementedError(
            f"{tokenizer_name} is not supported or Please use a valid tokenizer_name"
        )
    # TODO: need to set the special/pad tokens correctly
    return tokenizer
