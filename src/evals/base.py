class Evaluator:
    
    def __init__(self, eval_cfg, template_args, model, tokenizer, **kwargs):
        self.eval_cfg = eval_cfg
        self.template_args = template_args
        self.model = model
        self.tokenizer = tokenizer
        self.load_datasets()
        self.load_collators()
        self.load_dataloaders()
    
    def load_datasets(self):
        pass

    def load_collators(self):
        pass
    
    def load_dataloader(self):
        pass
    
    def evaluate(self):
        pass
        