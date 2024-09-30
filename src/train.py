import hydra
from omegaconf import DictConfig
from data import get_dataset
from model import get_model
from torch.utils.data import DataLoader
from data.utils import DataCollatorForSupervisedDataset
from trainer import load_trainer_args, load_trainer
import sys

@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    model_cfg = cfg.model
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)
    
    # Load Dataset
    tokenizer_details = {'tokenizer': tokenizer, 'template_cfg': model_cfg.chat_templating}
    dataset = get_dataset("TOFU_QA", cfg.data, tokenizer_details)
    

    collator = DataCollatorForSupervisedDataset(tokenizer)
    dataloader = DataLoader(dataset, collate_fn=collator, batch_size=3)
    batch = next(iter(dataloader))
    print(batch.keys())
    
    # # Get Trainer
    # trainer_cfg = cfg.get("trainer", None)
    # trainer_name = trainer_cfg.get("name", None)
    # assert trainer_cfg is not None, ValueError("Please set trainer")
    # trainer_args = load_trainer_args(trainer_cfg)
    # trainer = load_trainer(
    #     trainer_name=trainer_name,
    #     trainer_args=trainer_args,
    #     model=model,
    #     train_dataset=dataset,
    #     eval_dataset=dataset,
    #     tokenizer=tokenizer,
    #     # data_collator=None,
    #     data_collator=DataCollatorForSupervisedDataset,
    # )

    # trainer_args.do_train = True
    # trainer_args.do_eval = True
    
    # if trainer_args.do_train:
    #     trainer.train()

    # if trainer_args.do_eval:
    #     trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
