from typing import Callable, Any, Dict
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
        self.pre_compute_metrics: Dict[str, Callable] = {}

    def get_datasets(self, dataset_cfgs=None, **kwargs):
        """Load the datasets from config"""
        if self.data:
            return self.data
        data = get_datasets(
            tokenizer=kwargs.get("tokenizer", None),
            template_args=kwargs.get("template_args", None),
            dataset_cfgs=dataset_cfgs,
        )
        return data

    def get_collators(self, collator_cfgs=None, **kwargs):
        """Load the collators from config"""
        if self.collators:
            return self.collators
        collators = get_collators(
            tokenizer=kwargs.get("tokenizer", None), collator_cfgs=collator_cfgs
        )
        return collators

    def set_pre_compute_metrics(self, metrics: Dict[str, Callable]):
        self.pre_compute_metrics.update(metrics)

    def evaluate_metric(self, model, metric_name, **kwargs):
        print(f"Evaluating {metric_name}")
        results = self._metric_fn(model, **kwargs)
        return results

    def prepare_kwargs_evaluate_metric(self, model, metric_name, cache={}, **kwargs):
        """Prepare the kwargs required to call the metric_fn defined by user.
        - Loads datasets, collators, results for pre_compute metrics
        Returns:
            Dict: Updated kwargs with datasets, collators, pre_compute results loaded
        """
        dataset_cfgs = kwargs.pop("datasets", None)
        if dataset_cfgs is not None:
            data = self.get_datasets(dataset_cfgs=dataset_cfgs, **kwargs)
            kwargs.update({"data": data})

        collator_cfgs = kwargs.pop("collators", None)
        if collator_cfgs is not None:
            collators = self.get_collators(collator_cfgs=collator_cfgs, **kwargs)
            kwargs.update({"collators": collators})

        pre_compute_cfgs = kwargs.pop("pre_compute", {})
        pre_metric_results = {}
        for pre_metric_name, pre_metric_cfg in pre_compute_cfgs.items():
            access_name = pre_metric_cfg.get("access_key", pre_metric_name)
            _results = {}
            if pre_metric_name in cache:
                print(
                    f"Skipping {metric_name}'s precompute {pre_metric_name}, already evaluated."
                )
                _results = cache[pre_metric_name]
            else:
                pre_metric = self.pre_compute_metrics.get(pre_metric_name, None)
                assert pre_metric is not None, ValueError(
                    f"No pre_compute metric of name {pre_metric_name}"
                )
                pre_metric_kwargs = kwargs.copy()
                pre_metric_kwargs.update(**pre_metric_cfg)
                _results = pre_metric.evaluate(
                    model, pre_metric_name, cache=cache, **pre_metric_kwargs
                )
            pre_metric_results.update({access_name: _results})
        kwargs.update({"pre_compute": pre_metric_results})
        return kwargs

    def evaluate(self, model, metric_name, cache, **kwargs):
        """Evaluates a metric including its pre_compute metrics"""
        if metric_name in cache:
            print(f"Skipping {metric_name}, already evaluated.")

        metric_kwargs = self.prepare_kwargs_evaluate_metric(
            model, metric_name, cache, **kwargs
        )
        results = self.evaluate_metric(model, metric_name, **metric_kwargs)
        cache.update({metric_name: results})
        return results

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
