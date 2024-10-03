import hydra
from omegaconf import DictConfig
from data import get_datasets, get_collators
from model import get_model, get_dtype
from trainer import load_trainer

@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    trainer_cfg = cfg.trainer
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    assert trainer_cfg is not None, ValueError("Trainer config not set")
    torch_dtype = get_dtype(trainer_args=trainer_cfg.args)
    model, tokenizer = get_model(model_cfg, torch_dtype)
    
    # Load Dataset
    data_cfg = cfg.data
    collator_cfg = cfg.collator
    dataset = get_datasets(data_cfg, tokenizer=tokenizer, template_args=template_args)
    eval_data_cfg = cfg.get("eval_data", None)
    if eval_data_cfg:
        eval_dataset = get_datasets(eval_data_cfg, tokenizer=tokenizer, template_args=template_args)
    collator = get_collators(collator_cfg, tokenizer=tokenizer)
    
    # Get Trainer
    # TODO: modify save directory in trainer_cfg.args.output_dir based on
    # cfg.model, cfg.trainer, cfg.data and any other arguments which change
    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator
    )

    if trainer_args.do_train:
        trainer.train()
        model.save_pretrained(trainer_args.output_dir)
        tokenizer.save_pretrained(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
