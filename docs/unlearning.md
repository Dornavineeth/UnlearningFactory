<div align="center">    
 
# Unlearning

</div>


## Preliminaries
Refer to the [Components Guide](/docs/components.md) for instructions on adding individual components such as datasets, models, and trainers.


## Quick Start

Example script for launching an unlearning process:

```bash
# you can set configs in the yaml files directly or override them as below
python src/train.py --config-name=unlearn.yaml \
# model to unlearn
model=Llama-3.1-8B-Instruct \
# Override and provide path to pre-unlearning finetuned model
model.model_args.pretrained_model_name_or_path=<LOCAL PATH> \
# Unlearning method
trainer=GradDiff \    
# Override alpha parameter
trainer.method_args.alpha=0.5 \
# Forget dataset
data.forget=TOFU_QA_forget \
# Retain dataset
data.retain=TOFU_QA_retain \
# Evaluation dataset for trainer
data.eval=TOFU_QA_forget_para \
# Collator for datasets
collator=DataCollatorForSupervisedDataset
```
- **--config-name=unlearn.yaml**: Specifies the top-level config [unlearn.yaml](../configs/unlearn.yaml) file that loads configurations for each component used in unlearning.
- **model=Llama-3.1-8B-Instruct**: Loads the model configuration from [Llama-3.1-8B-Instruct.yaml](../configs/model/Llama-3.1-8B-Instruct.yaml)
- **model.model_args.pretrained_model_name_or_path=LOCAL PATH**: Overrides the model path defined in [Llama-3.1-8B-Instruct.yaml](../configs/model/Llama-3.1-8B-Instruct.yaml).
- **trainer=GradDiff**: Loads the unlearning trainer [GradDiffTrainer](../src/trainer/unlearn/grad_diff.py) with the configuration defined in [GradDiff.yaml](../configs/trainer/GradDiff.yaml).
- **trainer.method_args.alpha=0.5**: Overrides the alpha parameter [GradDiff.yaml](../configs/trainer/GradDiff.yaml).
- **data.forget=TOFU_QA_forget**: Sets the forget dataset to load [QADataset](../src/data/tofu.py) with config [TOFU_QA_forget.yaml](../configs/data/datasets/TOFU_QA_forget.yaml) for unlearning.
- **data.retain=TOFU_QA_retain**: Sets the retain dataset for [QADataset](../src/data/tofu.py) with config [TOFU_QA_retain.yaml](../configs/data/datasets/TOFU_QA_retain.yaml) for unlearning.
- **data.eval=TOFU_QA_forget_para**: Sets the evaluation dataset for [QADataset](../src/data/tofu.py) with config [TOFU_QA_forget_para.yaml](../configs/data/datasets/TOFU_QA_forget_para.yaml) for unlearning.

## Add new Unlearning Method

We provide an [UnlearningTrainer](/src/trainer/unlearn/base.py) which is specifically designed for unlearning and can perform evaluation during unlearning.

__NOTE__: See [Adding a New Trainer](/docs/components.md#trainer) for instructions on adding a new `Trainer` to the repository.

### Unlearning Trainer Handler

To add a new unlearning method, implement new trainer handler extending [UnlearningTrainer](/src/trainer/unlearn/base.py) in [src/trainer/unlearn](/src/trainer/unlearn/).

```python
# /src/trainer/unlearn/custom_method.py

from trainer.unlearn.base import UnlearningTrainer

class CustomMethod(UnlearningTrainer):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)
        ...
    
    def compute_loss(self, model, inputs, return_outputs=False):
        ...
        return (loss, forget_outputs) if return_outputs else loss
```
### Register Trainer

To make a custom `CustomMethod` Trainer handler accessible within the system, resgister it in the [TRAINER_REGISTRY](../src/trainer/__init__.py).


```python
# src/trainer/__init__.py

from trainer/unlearn/custom_method import CustomMethodTrainer

_register_trainer(CustomMethod)
```


### Unlearning Trainer Config

Define the config/arguments for the `CustomMethod` trainer in [configs/trainer](../configs/trainer/).

Example config for above `CustomMethod` :

```yaml
handler: CustomMethod 
args: # CustomMethod
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 16
  gradient_accumulation_steps: 4
  warmup_steps: 5
  learning_rate: 1e-5
  bf16: True
  bf16_full_eval: True
  logging_steps: 5
  output_dir: ${paths.train_output_dir}
  logging_dir: ${trainer.args.output_dir}/logs
  optim: paged_adamw_32bit
  save_strategy: epoch
  save_only_model: True
  weight_decay: 0.01
  do_train: True
  do_eval: True
  eval_strategy: epoch
  num_train_epochs: 10
  seed: 0
```