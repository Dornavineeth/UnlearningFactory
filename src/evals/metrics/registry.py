
from typing import Callable, Any, Dict
from evals.metrics.base import UnlearningMetric

REGISTERED_METRICS: Dict[str, UnlearningMetric] = {}

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
        REGISTERED_METRICS[name] = UnlearningMetric(
            name=name, 
            data_cfg=self.data_cfg,
            collator_cfg=self.collator_cfg,
            metric_fn=metric_fn
        )
        return REGISTERED_METRICS[name]