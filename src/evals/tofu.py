import os
import json
from evals.base import Evaluator
from evals.metrics import get_metrics

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
        
    def evaluate(self, **kwargs): # TODO: have overwrite argument
        logs_filename = os.path.join(self.eval_cfg.output_dir, 'TOFU_EVAL.json')
        if os.path.exists(logs_filename):
            with open(logs_filename, 'r') as f:
                logs = json.load(f)
        else:
            logs = {}
        
        for metric_name, metric_fn in self.metrics.items():
            if metric_name in logs:
                print(f"Skipping {metric_name}, already evaluated.")
                continue
            print(f"Evaluating {metric_name}")
            kwargs = {
                'tokenizer':self.tokenizer,
                'template_args':self.template_args
            }
            metrics_args = self.eval_cfg.metrics_cfg[metric_name]
            results = metric_fn(self.model, **kwargs, **metrics_args)
            logs[metric_name] = results
        
        os.makedirs(self.eval_cfg.output_dir, exist_ok=True)
        with open(logs_filename, "w") as f:
            json.dump(logs, f, indent=4)