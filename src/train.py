import hydra
from omegaconf import DictConfig

from model import get_model, get_tokenizer
from trainer import load_trainer_args, load_trainer


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    # Load Model
    model_cfg = cfg.get("model", None)
    assert model_cfg is not None, ValueError("Please set model")
    model = get_model(model_cfg)

    # Load Tokenizer
    tokenizer_cfg = cfg.get("tokenizer", None)
    assert tokenizer_cfg is not None, ValueError("Please set tokenizer")
    tokenizer = get_tokenizer(tokenizer_cfg)

    # Load Dataset
    # TODO
    from datasets import load_dataset

    dataset = load_dataset("locuslab/TOFU", "full")

    # Get Trainer
    trainer_cfg = cfg.get("trainer", None)
    trainer_name = trainer_cfg.get("name", None)
    assert trainer_cfg is not None, ValueError("Please set trainer")
    trainer_args = load_trainer_args(trainer_cfg)
    trainer = load_trainer(
        trainer_name=trainer_name,
        trainer_args=trainer_args,
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=None,
    )

    if trainer_args.do_train:
        trainer.train()

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
