# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

from typing import Dict, List, Optional, Union

import os
import logging
from transformers import Trainer
from torch.utils.data import Dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

logger = logging.getLogger(__name__)


class FinetuneTrainer(Trainer):
    def __init__(self, evaluator=None, template_args=None, *args, **kwargs):
        self.evaluator = evaluator
        self.template_args = template_args
        super().__init__(*args, **kwargs)

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def _evaluate(self, trial, ignore_keys_for_eval, skip_scheduler=False):
        """Performs Unlearning Evaluation pipeline and standard evaluation of HF Trainer"""
        if self.accelerator.is_local_main_process:
            if self.accelerator.num_processes == 1:
                run_dir = self._get_output_dir(trial=trial)
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                output_dir = os.path.join(run_dir, checkpoint_folder, "evals")
                os.makedirs(output_dir, exist_ok=True)
                eval_args = {
                    "output_dir": output_dir,
                    "template_args": self.template_args,
                    "model": self.model,
                    "tokenizer": self.tokenizer,
                }
                self.evaluator.evaluate(**eval_args)
            else:
                logger.warning(
                    "Unlearning Evaluation skipped as it supports only single gpu configuarations"
                )
        return super()._evaluate(trial, ignore_keys_for_eval, skip_scheduler)
