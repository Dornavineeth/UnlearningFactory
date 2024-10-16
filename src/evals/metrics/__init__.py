from omegaconf import DictConfig
from evals.metrics.memorization import (
    qa_prob,
    qa_paraphrased_prob,
    qa_perturbed_prob,
    qa_rouge,
    qa_paraphrased_rouge,
    qa_perturbed_rouge,
    qa_bio_prob,
    qa_bio_rouge,
)

METRICS_REGISTRY: Dict[str, "UnlearningMetric"] = {}

def _register_metric(metric_class):
    METRICS_REGISTRY[metric_class.name] = metric

def _get_single_metric(metric_name, metric_cfg, **kwargs):
    metric = METRICS_REGISTRY.get(metric_name)
    if metric is None:
        raise NotImplementedError(f"{metric_name} not implemented")
    else:
        return metric

def get_metrics(metric_cfgs: DictConfig, **kwargs):
    metrics = {}
    for metric_name, metric_cfg in metric_cfgs.items():
        metrics[metric_name] = _get_single_metric(metric_name, metric_cfg, **kwargs)
    return metrics

_register_metric(qa_prob)
_register_metric(qa_perturbed_prob)
_register_metric(qa_paraphrased_prob)
_register_metric(qa_rouge)
_register_metric(qa_paraphrased_rouge)
_register_metric(qa_perturbed_rouge)