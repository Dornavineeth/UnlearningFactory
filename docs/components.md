## Components

The OpenUnlearning framework requires a structured addition for new variants in the components of the unlearning pipeline. 

These are the components:
- **Trainer**: the algorithm used in LLM training or unlearning
- **Dataset**: a Dataset class for preprocessing raw data
- **Evaluation Metric**: a metric class that implements a specific evaluation of a model on a dataset
- **Benchmark**: an evaluation suite made of a set of evaluation metrics
- **Model**: the LLM used in unlearning
- **Collator**: data collation logic

For each component, the addition of a new variant involves three steps:
1. **Implement a handler**: Define the main code functionality for each component in a `.py` file.
2. **Register a handler**: Register a handler to access the functionality within training, unlearning, and evaluation pipelines.
3. **Define a configuration**: Define a `.yaml` config file (found in [`configs`](configs)) specifying the parameters required to run the implemented handler. We use Hydra extensively to set model, dataset and trainer configs, and also experimental configurations. 

__Note__: To add a model, you can skip steps 1 and 2 and just add a config file in [`configs/model`](configs/model) ([example](configs/model/Llama-2-7b-chat-hf.yaml)).




## Dataset 

Adding a new `Dataset`: 

- **Implement a handler**: Dataset handler code is implemented in the [`src/data`](../src/data/) directory.
- **Register the handler**: Handlers should be registered in the [`DATASET_REGISTRY`](../src/data/__init__.py).
- **Define the configuration**: Dataset configs are located in the [`configs/data/datasets`](../configs/data/datasets/) directory.


### Dataset Handler

Implement your own class extending `torch.utils.data.Dataset` in any file of [src/data](../configs/src/data/)


```python
class QADataset(Dataset):
    def __init__(
        self, hf_args,
        template_args, # will be pipelined from model
        tokenizer, # will be pipelined from model
        question_key="question", answer_key="answer", max_length=512
    ):
      ...

    def __getitem__(self, idx):
      ...
      return item
```

<!-- __Note__: `template_args` and `tokenizer` are additionally pipelined for the dataset object from model args for packaging and tokenizing the dataset. See [here](../src/train.py).  -->
<!-- I don't see why that line is necessary -->

### Register Dataset Handler


To make the dataset handler accessible, register its name in [`DATASET_REGISTRY`](../src/data/__init__.py), linking the handler and configuration.

Example: registering [`QADataset`](../src/data/tofu.py) (from `src/data/__init__.py`)
```python
from data.tofu import QADataset
_register_data(QADataset)
```
- `QADataset`: Implements the dataset handler which is used for training and evaluation.
- `_register_data`: Registers `QADataset`.

### Dataset Config

Specify the dataset handler arguments in the `.yaml` configuration file located in [`configs/data/datasets`](../configs/src/data/datasets).


Here’s an example configuration for a dataset named `TOFU_QA_full`:

```yaml
TOFU_QA_full: # data.tofu.QADataset
  handler: QADataset
  args:
    hf_args: # load_dataset from hf_hub
      name: "full"
      split: "train"
      path: "locuslab/TOFU"

    question_key: "question"
    answer_key: "answer"
    max_length: 512
```


## Trainer

Adding a new `Trainer`: 

- **Implement a handler**: Trainer handlers, which implement custom training algorithms, are implemented in the [`src/trainer`](../src/trainer/) directory.
- **Register the handler**: Register it in the [`TRAINER_REGISTRY`](../src/trainer/__init__.py).
- **Define the configuration**: Trainer configurations to run the handler are defined in the [`configs/trainer`](../configs/trainer/) directory.

A trainers can be defined for finetuning and also various unlearning methods.

### Trainer Handler

We extend HuggingFace's [`Trainer`](https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py) for our trainers.

###  Register Trainer Handler and Config

To make a custom Trainer accessible , register any custom `Trainer` handler implemented in the [`TRAINER_REGISTRY`](../src/trainer/__init__.py), linking the handler and configuration.


Example: registering trainer [`finetune`](./src/data/tofu.py) for finetuning and [`GradAscent`](../src/trainer/unlearn/grad_ascent.py) for unlearning (from `src/trainer/__init__.py`).
```python
from transformers import Trainer
from trainer.unlearn.grad_ascent import GradAscent
_register_trainer(Trainer)
_register_trainer(GradAscent)
```

### Trainer Config

Here’s an example configuration for finetuning using [HuggingFace Trainer](https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py) with its arguments.

```yaml
handler: Trainer # Train/finetune
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
- **args**: Contains the configuration parameters for `transformers.Trainer`, such as batch sizes, gradient accumulation, learning rate etc.



## Model

We support for loading [HuggingFace](https://huggingface.co/models) models using configs defined in the [`configs/models`](../configs/model/) directory. Here, we don't need to define or register handles as we use the default model and tokenizer handlers: `AutoModelForCausalLM` and `AutoTokenizer`. Model config files are accessed via their names in the main experiment config files (e.g.[here](../configs/train.yaml)).

Example config file for LLama3.1 Instruct model:
```yaml
model_args: # for AutoModelForCausalLM
  pretrained_model_name_or_path: "meta-llama/Llama-3.1-8B-Instruct"
  attn_implementation: 'flash_attention_2'
  torch_dtype: bfloat16

tokenizer_args: # for AutoTokenizer
  pretrained_model_name_or_path: "meta-llama/Llama-3.1-8B-Instruct"

template_args: # used in src/data/utils.py
  apply_chat_template: False # the below lines aren't needed when apply_chat_template is True
  system_prompt: You are a helpful assistant.
  system_prompt_with_special_tokens: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"
  user_start_tag: "<|start_header_id|>user<|end_header_id|>\n\n"
  user_end_tag: "<|eot_id|>"
  asst_start_tag: "<|start_header_id|>assistant<|end_header_id|>\n\n"
  asst_end_tag: "<|eot_id|>"
```

- `model_args` are arguments required by [AutoModelForCausalLM](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForCausalLM).

- `tokenizer_args` are arguments required by [AutoTokenizer](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer).

- `template_args` can include any arguments which can be used to process the datasets before passing to model. For example, see [package_prompt_response](../src/data/utils.py)