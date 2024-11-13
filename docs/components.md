## Components

Our pipeline architecture is designed with modularity and flexibility in mind, allowing you to easily add and customize core components for training and fine-tuning. The major components in any pipeline we support include:

- **Datasets**
- **Trainer**
- **Collator**
- **Model**

For each component, the setup involves three steps: (except for setting models up, for which writing a config file is enough)
1. **Implement a handler**: Define the main functionality for each component in a `.py` file.
2. **Register a handler**: Register a handler to access the functionality within training, unlearning, and evaluation pipelines.
3. **Define a configuration**: Define a `.yaml` configuration file specifying the parameters required to run the implemented handler.

We use Hydra for config management to enable flexible, hierarchical, and dynamic [configuration management](/configs/).


## Dataset

We support multiple datasets utilized for each benchmark, with their [configuration](../configs/data/datasets/)  and [handler](../src/data/). 

Adding a new `Dataset` involves: 

- **Implementing a handler**: Dataset handlers are implemented in the [src/data](../src/data/) directory.
- **Registering the handler**: Handlers should be registered in the [DATASET_REGISTRY](../src/data/__init__.py).
- **Defining the configuration**: Dataset configurations are located in the [configs/data/datasets](../configs/data/datasets/) directory.


### Dataset Handler

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

__NOTE__: `template_args` and `tokenizer` are additionally pipelined for the dataset object from model args for packaging and tokenizing the dataset. See [here](../src/train.py). 


### Dataset Config

Specify the dataset handler arguments in the `.yaml` configuration file located in [configs/data/datasets](../configs/src/data/datasets).


Here’s an example configuration for a dataset named `TOFU_QA_FULL`:

```yaml
TOFU_QA_FULL: # data.tofu.QADataset
  handler: QADataset
  args:
    hf_args: # load_dataset
      name: "full"
      split: "train"
      path: "locuslab/TOFU"

    question_key: "question"
    answer_key: "answer"
    max_length: 512
```

### Register Dataset Handler


To make the dataset handler accessible, register its name in [DATASET_REGISTRY](../src/data/__init__.py), linking the handler and configuration.

Example of registering [QADataset](../src/data/tofu.py)
```python
# src/data/__init__.py
from data.tofu import QADataset

_register_data(QADataset)
```

In this example:
- `QADataset`: Implements the dataset handler which is used for training and evaluation.
- `_register_data`: Registers `QADataset`.



## Trainer

Adding a new `Trainer` involves: 

- **Implementing a handler**: Trainer handlers, which implement custom training behaviors, are implemented in the [src/trainer](../src/trainer/) directory.
- **Registering the handler**: Register it in the [TRAINER_REGISTRY](../src/trainer/__init__.py).
- **Defining the configuration**: Trainer configurations to run the handler are defined in the [configs/trainer](../configs/trainer/) directory.

A trainers can be defined for finetuning and also various unlearning methods.

### Trainer Handler

We build on Hugging Face's [`Trainer`](https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py) for fine-tuning.
To add a custom trainer one could extend the `Trainer` class.

###  Register Trainer Handler and Config

To make a custom Trainer accessible within the system, register the `Trainer` or any custom `Trainer` handler implemented in the [TRAINER_REGISTRY](../src/trainer/__init__.py).


Example of registering trainers: [finetune](./src/data/tofu.py) for finetuning and [GradAscent](../src/trainer/unlearn/grad_ascent.py) for unlearning.
```python
from transformers import Trainer
from trainer.unlearn.grad_ascent import GradAscent # GradAscent Unlearning method.

# Register Finetuning Trainer
_register_trainer(Trainer)
_register_trainer(GradAscent)
```

### Trainer Config

Here’s an example configuration for finetuning using [Hugging Face Trainer](https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py) with its arguments.

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
- **args**: Contains the configuration parameters for `transformers.Trainer`, such as batch sizes, gradient accumulation, learning rate, and logging preferences.



## Model

We provide support for loading models from [Hugging Face](https://huggingface.co/models). All model configurations are defined in the [configs/models](../configs/model/) directory. Here the default model and tokenizer handlers are assumed to be `AutoModelForCausalLM` and `AutoTokenizer` of [Hugging Face](https://huggingface.co/models). These config files are accessed via their names in main experiment config files (e.g.[here](../configs/train.yaml)).

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

__NOTE__ 
1. `model_args` and `tokenizer_args` should only involve arguments which `AutoModelForCausalLM` and `AutoTokenizer` accept.
2. Before you attempt to use a HuggingFace model, ensure you have access to it. Models, especially from the Llama family are usually gated and require one to submit an access request. In such cases, go to the model page and [follow these instructions](https://huggingface.co/docs/hub/en/models-gated#access-gated-models-as-a-user).