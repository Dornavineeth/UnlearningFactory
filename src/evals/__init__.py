from typing import Dict, Any
from omegaconf import DictConfig
from evals.tofu import TOFUEvaluator

EVALUATOR_REGISTRY: Dict[str, Any] = {}


def _register_evaluator(evaluator_name, evaluator_class):
    EVALUATOR_REGISTRY[evaluator_name] = evaluator_class


def get_evaluator(name: str, eval_cfg: DictConfig, **kwargs):
    evaluator = EVALUATOR_REGISTRY.get(name)
    if evaluator is None:
        raise NotImplementedError(f"{name} not implemented or not registered")
    return evaluator(eval_cfg, **kwargs)


def get_evaluators(eval_cfgs: DictConfig, **kwargs):
    evaluators = {}
    for eval_name, eval_cfg in eval_cfgs.items():
        evaluators[eval_name] = get_evaluator(eval_name, eval_cfg, **kwargs)
    return evaluators


_register_evaluator("tofu", TOFUEvaluator)
