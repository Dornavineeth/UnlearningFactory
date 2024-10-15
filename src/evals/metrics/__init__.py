from omegaconf import DictConfig

from evals.metrics.registry import REGISTERED_METRICS
from evals.metrics.memorization import (
    qa_prob,
    qa_paraphrased_prob,
    qa_perturbed_prob,
    qa_rouge,
    qa_paraphrased_rouge,
    qa_perturbed_rouge
) 
# need to import so that the class is initialised, thus populating the dict
# but results in unused imports - not sure how to
# also results in having to edit the init file

def _get_single_metric(metric_name, metric_cfg, **kwargs):
    metric = REGISTERED_METRICS.get(metric_name)
    if metric is None:
        raise NotImplementedError(f"{metric_name} not implemented")
    else:
        return metric

def get_metrics(metric_cfgs: DictConfig, **kwargs):
    metrics = {}
    for metric_name, metric_cfg in metric_cfgs.items():
        metrics[metric_name] = _get_single_metric(metric_name, metric_cfg, **kwargs)
    
    return metrics