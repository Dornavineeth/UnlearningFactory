from omegaconf import DictConfig
from .tofu import TOFUEvaluator

def get_evaluator(eval_cfg: DictConfig, **kwargs):
    name = eval_cfg.name
    if name == 'tofu':
        return TOFUEvaluator(eval_cfg, **kwargs)
    else:
        raise NotImplementedError