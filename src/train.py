import hydra
from omegaconf import DictConfig
from data import get_dataset
from model import get_model, get_config_from_model_name
from data.utils import DataCollatorForSupervisedDataset
from trainer import load_trainer_args, load_trainer


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    model, tokenizer = get_model(cfg.model)
    model_cfg = get_config_from_model_name(cfg.model)
    
    # Load Dataset
    model_details = {'tokenizer': tokenizer, 'template_cfg': model_cfg['chat_templating']}
    dataset_details = {'split': 'full', 'data_path': 'locuslab/TOFU', 'question_key':'question', 'answer_key':'answer'}
    dataset_details = {**dataset_details, **model_details}
    dataset = get_dataset("TOFU_QA", dataset_details)
    
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
        data_collator=DataCollatorForSupervisedDataset,
    )

    if trainer_args.do_train:
        trainer.train()

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
