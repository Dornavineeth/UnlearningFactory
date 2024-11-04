## Components

Our pipeline architecture is designed with modularity and flexibility in mind, allowing you to easily add and customize core components for training and fine-tuning. The major components in any pipeline we support include:

- **Model**
- **Datasets**
- **Trainer**
- **Collator**

Each component follows a consistent setup process:
1. **Implement the handler**: Define the main functionality for each component in a dedicated `.py` file.
2. **Define the configuration**: Specify the component’s settings in a `.yaml` configuration file.
3. **Register the component**: Link the handler and configuration by registering the component, making it accessible in the training/unlearning/evalaution pipelines.

We use Hydra for [config](/configs/) management to enable flexible, hierarchical, and dynamic configuration for easy experimentation.

## Model
We provide support for loading models from [Hugging Face](https://huggingface.co/models). All model configurations are defined in the [configs/models](../configs/model/) directory.

Example config file for LLama3.1 Instruct model:
```yaml
model_args: # AutoModelForCausalLM
  pretrained_model_name_or_path: "meta-llama/Llama-3.1-8B-Instruct" # replace to load local models
  attn_implementation: 'flash_attention_2'
  torch_dtype: bfloat16

tokenizer_args: # AutoTokenizer
  pretrained_model_name_or_path: "meta-llama/Llama-3.1-8B-Instruct"

template_args:
  apply_chat_template: False
  user_start_tag: "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
  user_end_tag: "<|eot_id|>"
  asst_tag: "<|start_header_id|>assistant<|end_header_id|>\n\n"
```

- `model_args` include arguments for the [AutoModelForCausalLM](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForCausalLM).

- `tokenizer_args` include arguments for [AutoTokenizer](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer).

- `template_args` can include any arguments which can be used to process the datasets before passing to model. For example, see [package_prompt_response](../src/data/utils.py)

__NOTE__ `model_args` and `tokenizer_args` shouldn't take any other arguments which `AutoModelForCausalLM` and `AutoTokenizer` doesnt accept respectively.

## Dataset

We support multiple datasets utilized for each benchmark, with their [configuration](../configs/data/datasets/)  and [handler](../src/data/). 

Adding new `Dataset` invloves implementing 

- **Handler**: Dataset handlers are implemented in the [src/data](../src/data/) directory.
- **Configuration**: Dataset configurations are located in the [configs/data/datasets](../configs/data/datasets/) directory.
- **Register**: Both configurations and handlers should be registered in the [DATASET_REGISTRY](../src/data/__init__.py).


## Dataset Handler

Implement your own class extending `torch.utils.data.Dataset` in any file of [src/data](../configs/src/data/)


```python
class QADataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args, # will be pipelined from model
        tokenizer, # will be pipelined from model
        question_key="question",
        answer_key="answer",
        max_length=512,
        predict_with_generate=False,
    ):
      ...

    def __getitem__(self, idx):
      ...
      return item
```

__NOTE__: `template_args` and `tokenizer` are additionally piplelined for any dataset for packaging and tokenizing. See [here](../src/train.py). 


## Dataset Config

Specify the dataset handler arguments in the `.yaml` configuration file located in [configs/data/datasets](../configs/src/data/datasets).


Here’s an example configuration for a dataset named `TOFU_QA_FULL`:

```yaml
TOFU_QA_FULL: # data.tofu.QADataset
  args:
    hf_args: # load_dataset
      name: "full"
      split: "train"
      path: "locuslab/TOFU"

    question_key: "question"
    answer_key: "answer"
    max_length: 512
```

### Register Dataset Handler and Config


To make the dataset accessible, register its name in the Dataset handler within [DATASET_REGISTRY](../src/data/__init__.py), linking the handler and configuration.



Example of registering [QADataset](../src/data/tofu.py)
```python
from data.tofu import QADataset

_register_data("TOFU_QA_FULL", QADataset)
```

In this example:
- `QADataset`: Implements the dataset handler which is used for training and evaluation.
- `_register_data`: Registers `TOFU_QA_FULL` with `QADataset`


## Trainer

Adding new `Trainer` inlcudes implementing

- **Handlers**: Trainer handlers, which implement custom training behaviors, are located in the [src/trainer](../src/trainer/) directory.
- **Configurations**: Trainer configurations are defined in the [configs/trainer](../configs/trainer/) directory.
- **Registern**: Both the trainer configurations and their respective handlers are registered in the [TRAINER_REGISTRY](../src/trainer/__init__.py).


### Trainer Handler

We provide a [Hugging Face's](https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py) `Trainer` for fine-tuning and training.
To add a custom trainer we could extend the `Trainer` class.



### Trainer Config

Here’s an example configuration for finetuning using [Hugging Face Trainer](https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py) with its arguments.

```yaml
name: finetune # Train/finetune
args: # transformers.Trainer
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
- **name**: Specifies the type of trainer (e.g., finetune).
- **args**: Contains the configuration parameters for `transformers.Trainer`, such as batch sizes, gradient accumulation, learning rate, and logging preferences.


###  Register Trainer Handler and Config

To make a custom Trainer accessible within the system, register its name with the `Trainer` handler in the [TRAINER_REGISTRY](../src/trainer/__init__.py).


Example of registering trainers [finetune][../src/data/tofu.py](https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py) for finetuning and [GradAscent](../src/trainer/unlearn/grad_ascent.py) for unlearning.
```python
from transformers import Trainer
from trainer.unlearn.grad_ascent import GradAscentTrainer

# Register Finetuning Trainer
_register_trainer("finetune", Trainer)
_register_trainer("GradAscent", GradAscentTrainer)
```

<!-- ## Metric -->
