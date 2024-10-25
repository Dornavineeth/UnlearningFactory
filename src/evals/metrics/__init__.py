from typing import Dict
from omegaconf import DictConfig
from evals.metrics.base import UnlearningMetric
from evals.metrics.memorization import (
    q_a_prob,
    q_para_a_para_prob,
    q_a_para_prob,
    q_a_pert_prob,
    q_a_rouge,
    q_para_a_para_rouge,
    q_a_para_rouge,
    q_a_pert_rouge,
    bio_prob,
    bio_rouge,
)

METRICS_REGISTRY: Dict[str, UnlearningMetric] = {}


def _register_metric(metric):
    METRICS_REGISTRY[metric.name] = metric


def _get_single_metric(metric_name, metric_cfg, **kwargs):
    metric = METRICS_REGISTRY.get(metric_name)
    if metric is None:
        raise NotImplementedError(f"{metric_name} not implemented")
    return metric


def get_metrics(metric_cfgs: DictConfig, **kwargs):
    metrics = {}
    for metric_name, metric_cfg in metric_cfgs.items():
        metrics[metric_name] = _get_single_metric(metric_name, metric_cfg, **kwargs)
    return metrics


_register_metric(q_a_prob)
_register_metric(q_a_para_prob)
_register_metric(q_a_pert_prob)
_register_metric(q_para_a_para_prob)
_register_metric(q_a_rouge)
_register_metric(q_para_a_para_rouge)
_register_metric(q_a_para_rouge)
_register_metric(q_a_pert_rouge)
_register_metric(bio_prob)
_register_metric(bio_rouge)