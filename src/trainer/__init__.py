from typing import Dict
from transformers import Trainer, TrainingArguments


def load_trainer_args(trainer_cfg: Dict, **kwargs):
    trainer_args = trainer_cfg.get("args", None)
    trainer_args = TrainingArguments(**trainer_args)
    return trainer_args


def load_trainer(
    trainer_name: str,
    trainer_args,
    model,
    train_dataset,
    eval_dataset,
    tokenizer,
    data_collator,
):
    if trainer_name in ["finetune"]:
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            args=trainer_args,
        )
    else:
        raise NotImplementedError(
            f"{trainer_name} is not supported or Please use a valid trainer_name"
        )
    return trainer
