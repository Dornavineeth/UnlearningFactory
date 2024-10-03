import hydra
from omegaconf import DictConfig
from model import get_model, get_dtype
# from data import get_datasets
from evals import get_evaluator

@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to evaluate models
    Args:
        cfg (DictConfig): Config to train
    """
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    eval_cfg = cfg.eval
    torch_dtype = get_dtype(dtype_str=eval_cfg.dtype)
    model, tokenizer = get_model(model_cfg, torch_dtype)
    
    evaluator = get_evaluator(eval_cfg, template_args=template_args, model=model, tokenizer=tokenizer)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
