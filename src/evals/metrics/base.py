from typing import Callable, Any
from omegaconf import DictConfig
from data import get_datasets, get_collators

class UnlearningMetric:
    
    def __init__(
        self,
        name: str,
        data_cfg: DictConfig,
        collator_cfg: DictConfig,
        metric_fn: Callable[..., Any],
    ):
        self.name = name
        self.data_cfg = data_cfg
        self.collator_cfg = collator_cfg
        self._metric_fn = metric_fn
        self.data = None
        self.collators = None

    def get_datasets(self, data_cfg: DictConfig, **kwargs):
        if self.data:
            return self.data
        data_kwargs = {
            'tokenizer': kwargs.get('tokenizer', None),
            'template_args': kwargs.get('template_args', None)
        }
        data = get_datasets(data_cfg, **data_kwargs)
        return data

    def get_collators(self, collator_cfg: DictConfig, **kwargs):
        if self.collators:
            return self.collators
        collator_kwargs = {
            'tokenizer': kwargs.get('tokenizer', None),
        }
        collators = get_collators(collator_cfg, **collator_kwargs)
        return collators

    def evaluate(self, model, **kwargs):
        data = self.get_datasets(self.data_cfg, **kwargs)
        collators = self.get_collators(self.collator_cfg, **kwargs)
        metric_kwargs = {
            "data": data,
            "collators": collators
        }
        return self._metric_fn(model, **metric_kwargs, **kwargs)
    
    def __call__(self, model, **kwargs):
        return self.evaluate(model, **kwargs)
    
    def __repr__(self) -> str:
        """Represents class object as string

        Returns:
            str: string representation of the class object
        """
        return f"{type(self).__name__} {self.name}"