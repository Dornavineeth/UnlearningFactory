from typing import Dict
from omegaconf import DictConfig
from evals.metrics.base import UnlearningMetric
from evals.metrics.memorization import probability, rouge, truth_ratio

METRICS_REGISTRY: Dict[str, UnlearningMetric] = {}


def _register_metric(metric):
    METRICS_REGISTRY[metric.name] = metric


def _get_single_metric(name: str, metric_cfg, **kwargs):
    metric_handler_name = metric_cfg.get("handler")
    assert metric_handler_name is not None, ValueError(f"{name} handler not set")
    metric = METRICS_REGISTRY.get(metric_handler_name)
    if metric is None:
        raise NotImplementedError(
            f"{metric_handler_name} not implemented or not registered"
        )
    pre_compute_cfg = metric_cfg.get("pre_compute", {})
    pre_compute_metrics = get_metrics(pre_compute_cfg, **kwargs)
    metric.set_pre_compute_metrics(pre_compute_metrics)
    return metric


def get_metrics(metric_cfgs: DictConfig, **kwargs):
    metrics = {}
    for metric_name, metric_cfg in metric_cfgs.items():
        metrics[metric_name] = _get_single_metric(metric_name, metric_cfg, **kwargs)
    return metrics


# Register Metrics for unlearning here
_register_metric(probability)
_register_metric(rouge)
_register_metric(truth_ratio)
