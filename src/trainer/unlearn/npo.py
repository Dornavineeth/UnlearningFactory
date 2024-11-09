import copy
from trainer.utils import compute_dpo_loss
from trainer.unlearn.grad_diff import GradDiff


class NPO(GradDiff):
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        if hasattr(self, "target_model"):
            self.target_model = copy.deepcopy(self.model).to("cuda")

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]

        forget_loss, forget_outputs = compute_dpo_loss(
            model=model,
            ref_model=self.target_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
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
