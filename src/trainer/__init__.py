from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments


def load_trainer_args(trainer_args: DictConfig, **kwargs):
    trainer_args = TrainingArguments(**trainer_args)
    return trainer_args


def load_trainer(
    trainer_cfg: DictConfig,
    model,
    train_dataset,
    eval_dataset,
    tokenizer,
    data_collator,
):
    trainer_name = trainer_cfg.name
    trainer_args = trainer_cfg.args
    trainer_args = load_trainer_args(trainer_args)
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
    return trainer, trainer_args
