from omegaconf import DictConfig
from evals.metrics.memorization import (
    qa_prob,
    qa_paraphrased_prob,
    qa_perturbed_prob,
    qa_rouge,
    qa_paraphrased_rouge,
    qa_perturbed_rouge
)
    
class unlearning_metric:
    
    def __init__(
        self,
        name: str,
        data_cfg,
        collator_cfg
    ):
        self.name = name
        self.data_cfg = data_cfg
        self.collator_cfg = collator_cfg
    
    
    def __call__(self,  metric_fn: Callable[..., Any]) -> UnlearningMetric:
        name = self.name or metric_fn.__name__
        return UnlearningMetric(
            name=name, 
            data_cfg=self.data_cfg,
            collator_cfg=self.collator_cfg,
            metric_fn=metric_fn
        )
        # METRICS_REGISTRY[name] = UnlearningMetric(
        #     name=name, 
        #     data_cfg=self.data_cfg,
        #     collator_cfg=self.collator_cfg,
        #     metric_fn=metric_fn
        # )
        # return METRICS_REGISTRY[name]

METRICS_REGISTRY: Dict[str, "UnlearningMetric"] = {}

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