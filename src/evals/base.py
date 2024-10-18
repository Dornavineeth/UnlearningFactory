class Evaluator:
    def __init__(self, eval_cfg, template_args, model, tokenizer, **kwargs):
        self.eval_cfg = eval_cfg
        self.template_args = template_args
        self.model = model
        self.tokenizer = tokenizer
        self.prepare_model()
        self.load_metrics()

    def prepare_model(self):
        pass

    def load_metrics(self):
        pass

    def evaluate(self):
        pass
