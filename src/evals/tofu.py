import os
import json
from .base import Evaluator
from .metrics import get_metrics

class TOFUEvaluator(Evaluator):
    
    def __init__(self, eval_cfg, template_args, model, tokenizer, **kwargs):
        super().__init__(eval_cfg, template_args, model, tokenizer, **kwargs)
        self.logs = {}

    def prepare_model(self):
        self.device = self.eval_cfg.device
        self.model.to(self.device)
        self.model.eval()
    
    def load_metrics(self):
        self.metrics_cfg = self.eval_cfg.metrics_cfg
        self.metrics = get_metrics(self.metrics_cfg)
        
    def evaluate(self, **kwargs):
        logs = {}
        for metric_name, metric_fn in self.metrics.items():
            print(f"Evaluating {metric_name}")
            kwargs = {
                'tokenizer':self.tokenizer,
                'template_args':self.template_args
            }
            metrics_args = self.eval_cfg.metrics_cfg[metric_name]
            results = metric_fn(self.model, **kwargs, **metrics_args)
            logs[metric_name] = results
        os.makedirs(self.eval_cfg.output_dir, exist_ok=True)
            
        # Define your logs dictionary and the file path
        logs_filename = os.path.join(self.eval_cfg.output_dir, 'TOFU_EVAL.json')

        # Check if the file already exists
        if os.path.exists(logs_filename):
            # If it exists, read the existing content
            with open(logs_filename, "r") as f:
                existing_logs = json.load(f)  # Read the existing dictionary from the file
        else:
            existing_logs = {}  # Initialize an empty dictionary if the file doesn't exist

        # Merge the new logs with the existing logs (new logs will overwrite existing keys if they conflict)
        existing_logs.update(logs)  # Assuming `logs` is your new logs dictionary

        # Write the updated logs back to the file in a pretty-printed format
        with open(logs_filename, "w") as f:
            json.dump(existing_logs, f, indent=4)