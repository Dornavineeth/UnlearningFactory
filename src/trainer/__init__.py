from typing import Dict, Any
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments

from trainer.unlearn.grad_ascent import GradAscentTrainer
from trainer.unlearn.grad_diff import GradDiffTrainer
from trainer.unlearn.npo import NPOTrainer
from trainer.unlearn.dpo import DPOTrainer

TRAINER_REGISTRY: Dict[str, Any] = {}


def _register_trainer(trainer_name, trainer_class):
    TRAINER_REGISTRY[trainer_name] = trainer_class


def load_trainer_args(trainer_args: DictConfig):
    trainer_args = dict(trainer_args)
    trainer_args = TrainingArguments(**trainer_args)
    return trainer_args


def load_trainer(
    trainer_cfg: DictConfig,
    model,
    train_dataset=None,
    eval_dataset=None,
    tokenizer=None,
    data_collator=None,
):
    trainer_name = trainer_cfg.name
    trainer_args = trainer_cfg.args
    method_args = trainer_cfg.get("method_args", {})
    trainer_args = load_trainer_args(trainer_args)
    trainer_cls = TRAINER_REGISTRY.get(trainer_name, None)
    assert trainer_cls is not None, NotImplementedError(
        f"{trainer_name} is not supported or Please use a valid trainer_name"
    )
    trainer = trainer_cls(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=trainer_args,
        **method_args,
    )
    return trainer, trainer_args


# Register Finetuning Trainer
_register_trainer("finetune", Trainer)

# Register Unlearning Trainer
_register_trainer("GradAscent", GradAscentTrainer)
_register_trainer("GradDiff", GradDiffTrainer)
_register_trainer("NPO", NPOTrainer)
_register_trainer("DPO", DPOTrainer)
