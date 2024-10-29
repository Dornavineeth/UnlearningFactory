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

    def prepare_model(self):
        self.device = self.eval_cfg.device
        self.model.to(self.device)
        self.model.eval()

    def load_metrics(self):
        self.metrics_cfg = self.eval_cfg.metrics_cfg
        self.metrics = get_metrics(self.metrics_cfg)

    def evaluate(self, overwrite=False, **kwargs):
        logs_filename = os.path.join(self.eval_cfg.output_dir, f"{self.name}_EVAL.json")
        logs = {}

        if os.path.exists(logs_filename):
            with open(logs_filename, "r") as f:
                logs = json.load(f)

        for metric_name, metric_fn in self.metrics.items():
            if not overwrite and metric_name in logs:
                print(f"Skipping {metric_name}, already evaluated.")
                continue
            print(f"Evaluating {metric_name}")
            kwargs = {"tokenizer": self.tokenizer, "template_args": self.template_args}
            metrics_args = self.eval_cfg.metrics_cfg[metric_name]
            results = metric_fn(self.model, **kwargs, **metrics_args)
                
            logs[metric_name] = results

            os.makedirs(self.eval_cfg.output_dir, exist_ok=True)
            with open(logs_filename, "w") as f:
                json.dump(logs, f, indent=4)