<div align="center">    
 
# Unlearning

</div>


## Preliminaries
Refer to the [Components Guide](/docs/components.md) for instructions on adding individual components such as datasets, models, and trainers.


## Quick Start

Example script for launching an unlearning process:

```bash
python src/train.py --config-name=unlearn.yaml \
model=Llama-3.1-8B-Instruct \ # model to unlearn
model.model_args.pretrained_model_name_or_path=<LOCAL PATH> \ # Override path to load model
trainer=GradDiff \ # unlearning method
trainer.method_args.alpha=0.5 \ # Override alpha 
data.forget=TOFU_QA_FORGET10 \ # forget dataset
data.retain=TOFU_QA_RETAIN90 \ # retain dataset
data.eval=TOFU_QA_FORGET10_P \ # evaluation dataset for trainer
collator=DataCollatorForSupervisedDataset # collator for datasets
```
- **--config-name=unlearn.yaml**: Specifies the top-level config [unlearn.yaml](../configs/unlearn.yaml) file that loads configurations for each component used in unlearning.
- **model=Llama-3.1-8B-Instruct**: Loads the model configuration from [Llama-3.1-8B-Instruct.yaml](../configs/model/Llama-3.1-8B-Instruct.yaml)
- **model.model_args.pretrained_model_name_or_path=LOCAL PATH**: Overrides the model path defined in [Llama-3.1-8B-Instruct.yaml](../configs/model/Llama-3.1-8B-Instruct.yaml).
- **trainer=GradDiff**: Loads the unlearning trainer [GradDiffTrainer](../src/trainer/unlearn/grad_diff.py) with the configuration defined in [GradDiff.yaml](../configs/trainer/GradDiff.yaml).
- **trainer.method_args.alpha=0.5**: Overrides the alpha parameter [GradDiff.yaml](../configs/trainer/GradDiff.yaml).
- **data.forget=TOFU_QA_FORGET10**: Sets the forget dataset to load [QADataset](../src/data/tofu.py) with config [TOFU_QA_FORGET10.yaml](../configs/data/datasets/TOFU_QA_FORGET10.yaml) for unlearning.
- **data.retain=TOFU_QA_RETAIN90**: Sets the retain dataset for [QADataset](../src/data/tofu.py) with config [TOFU_QA_RETAIN90.yaml](../configs/data/datasets/TOFU_QA_RETAIN90.yaml) for unlearning.
- **data.eval=TOFU_QA_FORGET10_P**: Sets the evaluation dataset for [QADataset](../src/data/tofu.py) with config [TOFU_QA_FORGET10_P.yaml](../configs/data/datasets/TOFU_QA_FORGET10_P.yaml) for unlearning.

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
# src/traianer/__init__.py

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