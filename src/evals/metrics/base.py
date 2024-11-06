from typing import Callable, Any
from data import get_datasets, get_collators


class UnlearningMetric:
    def __init__(
        self,
        name: str,
        metric_fn: Callable[..., Any],
    ):
        self.name = name
        self._metric_fn = metric_fn
        self.data = None
        self.collators = None
        self.pre_compute = {}

    def get_datasets(self, dataset_cfgs=None, **kwargs):
        if self.data:
            return self.data
        data = get_datasets(
            tokenizer=kwargs.get("tokenizer", None),
            template_args=kwargs.get("template_args", None),
            dataset_cfgs=dataset_cfgs,
        )
        return data

    def get_collators(self, collator_cfgs=None, **kwargs):
        if self.collators:
            return self.collators
        collators = get_collators(
            tokenizer=kwargs.get("tokenizer", None), collator_cfgs=collator_cfgs
        )
        return collators
    
    def pre_compute_metrics(self, model, metric_cfgs, cache, **kwargs):
        results = {}
        for metric_name, metric_cfg in  metric_cfgs.items():
            if metric_name in cache:
                print(f"Skipping {self.name}'s precompute {metric_name}, already evaluated.")
                metric_results = cache[metric_name]
            else:
                metric = self.pre_compute.get(metric_name, None)
                assert metric is not None, ValueError(f"No pre_compute metric of name {metric_name}")
                metric_results = metric.evaluate(model, cache=cache, **metric_cfg, **kwargs)
            results.update({metric_name: metric_results})
        return results

    def evaluate(self, model, cache, **kwargs):
        metric_kwargs = {}
        pre_compute_cfgs = kwargs.pop("pre_compute", {})
        pre_compute_results = self.pre_compute_metrics(model, pre_compute_cfgs, cache=cache, **kwargs)
        metric_kwargs.update({"pre_compute": pre_compute_results})
        dataset_cfgs = kwargs.pop("datasets", None)
        if dataset_cfgs is not None:
            data = self.get_datasets(dataset_cfgs=dataset_cfgs, **kwargs)
            metric_kwargs.update({"data": data})
        collator_cfgs = kwargs.pop("collators", None)
        if collator_cfgs is not None:
            collators = self.get_collators(collator_cfgs=collator_cfgs, **kwargs)
            metric_kwargs.update({"collators": collators})
        print(f"Evaluating {self.name}")
        results = self._metric_fn(model, **metric_kwargs, **kwargs)
        cache.update({self.name: results})
        return  results

    def __call__(self, model, **kwargs):
        return self.evaluate(model, **kwargs)

    def __repr__(self) -> str:
        """Represents class object as string

        Returns:
            str: string representation of the class object
        """
        return f"{type(self).__name__} {self.name}"


# decorator that wraps simple user-defined metric python functions
# into callable UnlearningMetric classes
class unlearning_metric:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, metric_fn: Callable[..., Any]) -> UnlearningMetric:
        name = self.name or metric_fn.__name__
        return UnlearningMetric(name=name, metric_fn=metric_fn)
