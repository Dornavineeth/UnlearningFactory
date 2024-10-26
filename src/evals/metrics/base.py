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
        self.collator = None

    def get_datasets(self, **kwargs):
        if self.data:
            return self.data
        data = get_datasets(tokenizer=kwargs.get("tokenizer", None),
                            template_args=kwargs.get("template_args", None),
                            data_cfgs=kwargs.get("data_cfg", None))
        return data

    def get_collators(self, **kwargs):
        if self.collator:
            return self.collator
        collator = get_collators(tokenizer=kwargs.get("tokenizer", None),
                                  collator_cfgs=kwargs.get("collator_cfg", None))
        return collator

    def evaluate(self, model, **kwargs):
        data = self.get_datasets(**kwargs) # returns only a single collator
        collator = self.get_collators(**kwargs) # returns only a single collator
        metric_kwargs = {"data": data, "collator": collator}
        return self._metric_fn(model, **metric_kwargs, **kwargs)

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
