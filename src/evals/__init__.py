from omegaconf import DictConfig
from evals.tofu import TOFUEvaluator

def get_evaluator(name: str, eval_cfg: DictConfig, **kwargs):
    if name == 'tofu':
        return TOFUEvaluator(eval_cfg, **kwargs)
    else:
        raise NotImplementedError

def get_evaluators(eval_cfgs: DictConfig, **kwargs):
    evaluators = {}
    for eval_name, eval_cfg in eval_cfgs.items():
        evaluators[eval_name] = get_evaluator(eval_name, eval_cfg, **kwargs)
    return evaluators