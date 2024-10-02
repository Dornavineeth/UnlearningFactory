from .base import Evaluator
from torch.utils.data import DataLoader
from data import get_datasets, get_collators


class TOFUEvaluator(Evaluator):
    
    def __init__(self, eval_cfg, template_args, model, tokenizer, **kwargs):
        super().__init__(eval_cfg, template_args, model, tokenizer, **kwargs)
        self.logs = {}
    
    def load_datasets(self):
        self.data_cfg = self.eval_cfg.data
        kwargs ={
            "template_args": self.template_args,
            "tokenizer": self.tokenizer
        }
        self.data = get_datasets(self.data_cfg, **kwargs)
        self.forget10_qa_key = "TOFU_QA_FORGET10"
        self.forget10_qa_p_key = "TOFU_QA_FORGET10_P"
        self.forget10_qa = self.data[self.forget10_qa_key]
        self.forget10_qa_p = self.data[self.forget10_qa_p_key]
        kwargs_gen = kwargs.copy()
        kwargs_gen['predict_with_generate'] = True
        self.data_gen = get_datasets(self.data_cfg, **kwargs_gen)
        self.forget10_qa_gen = self.data_gen[self.forget10_qa_key]
        self.forget10_qa_p_gen = self.data_gen[self.forget10_qa_p_key]
    
    def load_collators(self):
        self.collator_cfg = self.eval_cfg.collator
        kwargs = {
            "tokenizer": self.tokenizer
        }
        self.collator = get_collators(self.collator_cfg, **kwargs)
        kwargs_gen = kwargs.copy()
        kwargs_gen['padding_side'] = 'left'
        self.collator_gen = get_collators(self.collator_cfg, **kwargs_gen)
    
    def load_dataloaders(self):
        batch_size = self.eval_cfg.batch_size
        self.forget10_qa_dataloader = DataLoader(
            self.forget10_qa, batch_size=batch_size, collate_fn=self.collator
        )
        self.forget10_qa_p_dataloader = DataLoader(
            self.forget10_qa_p, batch_size=batch_size, collate_fn=self.collator_gen
        )
        self.forget10_qa_gen_dataloader = DataLoader(
            self.forget10_qa_gen, batch_size=batch_size, collate_fn=self.collator_gen
        )
        self.forget10_qa_p_gen_dataloader = DataLoader(
            self.forget10_qa_p_gen, batch_size=batch_size, collate_fn=self.collator_gen
        )
    
    def evaluate_loss_batch(self, model, batch, **kwargs):
        output = model(**batch)
        import pdb;pdb.set_trace()
        
    def evaluate_loss(self, dataloader, **kwargs):
        id_to_loss = {}
        for batch in dataloader:
            loss = self.evaluate_loss_batch(self.model, batch, **kwargs)
        return 
        
    
    def evaluate(self):
        self.evaluate_loss(self.forget10_qa_dataloader)
        import pdb;pdb.set_trace()
        pass