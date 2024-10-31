from trainer.utils import compute_dpo_loss
from trainer.unlearn.grad_diff import GradDiffTrainer


class NPOTrainer(GradDiffTrainer):
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_labels = forget_inputs["labels"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
        }
        forget_loss, forget_outputs = compute_dpo_loss(
            model=model,
            ref_model=self.target_model,
            inputs=forget_inputs,
            lose_labels=forget_labels,
            beta=self.beta,
        )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
