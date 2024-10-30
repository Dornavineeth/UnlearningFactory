from evals.base import Evaluator


class TOFUEvaluator(Evaluator):
    def __init__(self, eval_cfg, template_args, model, tokenizer, **kwargs):
        super().__init__('TOFU', eval_cfg, template_args, model, tokenizer, **kwargs)
