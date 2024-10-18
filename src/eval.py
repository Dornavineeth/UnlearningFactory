import hydra
from omegaconf import DictConfig
from model import get_model

# from data import get_datasets
from evals import get_evaluators


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to evaluate models
    Args:
        cfg (DictConfig): Config to train
    """
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    eval_cfgs = cfg.eval
    evaluators = get_evaluators(
        eval_cfgs, template_args=template_args, model=model, tokenizer=tokenizer
    )
    for evaluator_name, evaluator in evaluators.items():
        eval_args = eval_cfgs[evaluator_name]
        evaluator.evaluate(**eval_args)


if __name__ == "__main__":
    main()
