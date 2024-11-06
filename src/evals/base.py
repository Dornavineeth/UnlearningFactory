import os
import json
from evals.metrics import get_metrics


class Evaluator:
    def __init__(self, name, eval_cfg, template_args, model, tokenizer, **kwargs):
        self.eval_cfg = eval_cfg
        self.template_args = template_args
        self.model = model
        self.tokenizer = tokenizer
        self.prepare_model()
        self.load_metrics()
        self.name = name
        self.logs = {}
        self.logs_filename = os.path.join(self.eval_cfg.output_dir, f"{self.name}_EVAL.json")
        if os.path.exists(self.logs_filename):
            with open(self.logs_filename, "r") as f:
                self.logs = json.load(f)

    def prepare_model(self):
        self.device = self.eval_cfg.device
        self.model.to(self.device)
        self.model.eval()

    def load_metrics(self):
        self.metrics_cfg = self.eval_cfg.metrics
        self.metrics = get_metrics(self.metrics_cfg)

    def evaluate(self, overwrite=False, **kwargs):
        for metric_name, metric_fn in self.metrics.items():
            if not overwrite and metric_name in self.logs:
                print(f"Skipping {metric_name}, already evaluated.")
                continue
            kwargs = {"tokenizer": self.tokenizer, "template_args": self.template_args}
            metrics_args = self.eval_cfg.metrics[metric_name]
            _ = metric_fn(self.model, cache=self.logs, **kwargs, **metrics_args)
            # for p_metric_name, p_results in pre_compute_results.items():
            #     self.logs[p_metric_name] = p_results
            # self.logs[metric_name] = results
            os.makedirs(self.eval_cfg.output_dir, exist_ok=True)
            with open(self.logs_filename, "w") as f:
                json.dump(self.logs, f, indent=4)
