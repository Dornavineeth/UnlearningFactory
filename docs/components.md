# Components  

The OpenUnlearning framework requires a structured approach for adding new components in the unlearning pipeline.

This process involves three main steps:
1. __Implementing a handler__: Define the core logic for the component. A single handler can be reused across multiple components. For example, a handler that computes the ROUGE score can support various evaluation metrics across multiple datasets.
2. __Registering the handler__: Add the handler to a registry that links it to a key, allowing access during execution through the config files.
3. __Adding a config file__:  Set up a configuration using Hydra that specifies the handler and relevant parameters. These configurations can then be passed directly as arguments when running Python scripts.

---

## Documentation on adding each type of component  

1. [Trainer](#trainer) - Algorithm used in LLM training or unlearning  
2. [Dataset](#dataset) - Dataset class for preprocessing raw data  
3. [Evaluation Metric](#evaluation-metric) - Metric class implementing model evaluation  
4. [Benchmark](#benchmark) - Suite combining multiple evaluation metrics  
5. [Model](#model) - LLM used in unlearning  
6. [Collator](#collator) - Handles data collation logic  
7. [Experiment](#experiment) - Combines components into a final experiment config  

---

## Trainer  

To add a new **Trainer**:  

### Implement a handler  
We extend HuggingFace's [`Trainer`](https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py) for for custom training algorithms. Trainer handlers are written in [`src/trainer`](../src/trainer/).  

Example: defining a gradient-difference based unlearning trainer.

```python
class GradDiff(UnlearnTrainer):
    def __init__(self, gamma, alpha, ...):
        ...
      
    def compute_loss(self, model, inputs, return_outputs=False):
        ...
```

### Register the trainer handler  
Register the handler to link the class to the configs via the class name in [`TRAINER_REGISTRY`](../src/trainer/__init__.py).

Example: Registering a fine-tuning trainer and `GradDiff` unlearning trainer 

```python
from transformers import FinetuneTrainer
from trainer.unlearn.grad_ascent import GradDiff
_register_trainer(FinetuneTrainer) # class defined in src/trainer/base.py
_register_trainer(GradDiff) # class defined in src/trainer/unlearn/grad_diff.py
```

### Add a trainer to configs  

Add a config that uses the new trainer and set parameters. Trainer configurations are in [`configs/trainer`](../configs/trainer/). Each config contains a handler that points to the defined trainer class and the arguments used to initialise the trainer.

Example: Config file ([`configs/trainer/GradDiff.yaml`](../configs/trainer/GradDiff.yaml)) for GradDiff.
```yaml
handler: GradDiff
args: # HuggingFace TrainingArguments
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 16
  gradient_accumulation_steps: 4
  learning_rate: 1e-5
  num_train_epochs: 10
method_args: # Your own method-specific arguments
  gamma: 1.0
  alpha: 1.0
  retain_loss_type: NLL
```

---

## Dataset  

To add a new dataset, we create a generic preprocessing handler and then configure it to create a dataset:  

### Implement a handler  
Extend `torch.utils.data.Dataset` to to create dataset handlers for loading and preprocessing data. These are written in [`src/data`](../src/data/). A new dataset would then instantiated by providing its parameters (dataset column, length etc) to an existing dataset handler.

Example: defining a `PretrainingDataset` dataset handler to load texts for pre-training style next token prediction.

```python
class PretrainingDataset(Dataset):
    def __init__(self, hf_args, text_key, max_length, ...):
        ...

    def __getitem__(self, idx):
        ...
        return item
```

### Register the dataset handler  
Register the handler to link the class to the configs via the class name in [`DATASET_REGISTRY`](../src/data/__init__.py).

Example: Registering `PretrainingDataset`  

```python
from data.pretraining import PretrainingDataset
_register_data(PretrainingDataset)
```

### Add a dataset to configs  
Add a specific dataset that uses the `PretrainingDataset` class format. Dataset configurations go in [`configs/data/datasets`](../configs/data/datasets/). Each config contains a handler that points to the defined dataset class and the arguments used to create the dataset.

Example: add a config file for the `MUSE_forget` and `MUSE_forget_sust` datasets using the `PretrainingDataset` handler
```yaml
MUSE_forget: # the name of a particular dataset instance
  handler: PretrainingDataset
  args:
    hf_args:
      path: "muse-bench/MUSE-News"
      name: "raw"
      split: "forget"
    text_key: "text"
    max_length: 2048

MUSE_forget_sust: # another dataset
  handler: PretrainingDataset
  args:
    hf_args:
      path: "muse-bench/MUSE-Books"
      name: "sust"
      split: "forget_1"
    text_key: "text"
    max_length: 2048
```
---

## Evaluation Metric  

To add a new evaluation metric, we create a handler with the metric computation logic and then configure it. More documentation on adding metrics is in [`docs/evaluation.md`](evaluation.md)

### Implement a handler  
Metric handlers are implemented in [`src/evals/metrics`](../src/evals/metrics/), where we define handlers that compute individual statistics and/or aggregated metrics over a dataset such as ROUGE scores, KS-tests etc.

Example: implementing the `rouge` handler

```python
# in src/evals/metrics/memorization.py
@unlearning_metric(name="rouge")
def rouge(model, **kwargs):
    """Calculate ROUGE metrics and return the aggregated value along with per-index scores."""
    # kwargs is populated on the basic of the metric configuration
    tokenizer = kwargs["tokenizer"]
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    generation_args = kwargs["generation_args"]
    ...
    return {
        "agg_value": np.mean(rouge_values),
        "value_by_index": scores_by_index,
    }

```

### Register the metric handler  
Register the handler to link the class to the configs via the class name in [`METRIC_REGISTRY`](../src/evals/metrics/__init__.py).

Example: Registering `rouge` handler  

```python
from evals.metrics.memorization import rouge
_register_metric(rouge)
```

### Add a metric to configs  
Metric configurations are in [`configs/eval/tofu_metrics`](../configs/eval/tofu_metrics/) and [`configs/eval/muse_metrics`](../configs/eval/muse_metrics/). These create individual evaluation metrics by providing the handler a specific dataset and other parameters.

Example: Creating the config for MUSE's `forget_verbmem_ROUGE` ([`configs/eval/muse_metrics/forget_knowmem_ROUGE.yaml`](../configs/eval/muse_metrics/forget_knowmem_ROUGE.yaml)). 

```yaml
# @package eval.muse.metrics.forget_verbmem_ROUGE
# NOTE: the above line is not a comment. See 
# https://hydra.cc/docs/upgrades/0.11_to_1.0/adding_a_package_directive/
# it ensures that the below attributes are found in the config path
# eval.muse.metrics.forget_verbmem_ROUGE in the final config
defaults: # fill up forget_verbmem_ROUGE's inputs' configs
  - ../../data/datasets@datasets: MUSE_forget_verbmem
  - ../../collator@collators: DataCollatorForSupervisedDatasetwithIndex
  - ../../generation@generation_args: default
handler: rouge # the handler we defined above
rouge_type: rougeL_f1
batch_size: 8
# override default parameters
datasets:
  MUSE_forget_verbmem:
    args:
      hf_args:
        path: muse-bench/MUSE-${eval.muse.data_split}
      predict_with_generate: True
collators:
  DataCollatorForSupervisedDataset: 
    args:
      padding_side: left # for generation
generation_args:
  max_new_tokens: 128
```

---

## Benchmark  

A benchmark (also called evaluator) is a collection of evaluation metrics defined above (e.g. TOFU, MUSE). To add a new benchmark:  

### Implement a handler  

In the handlers in [`src/evals`](../src/evals/), you can add code to: modify the collection, aggregation and reporting of the metrics computed, any pre-eval model preparation etc.

### Register the benchmark handler  
Register the benchmark to link the class to the configs via the class name in [`BENCHMARK_REGISTRY`](../src/evals/__init__.py).

Example: Registering TOFU benchmark  

```python
from evals.tofu import TOFUEvaluator
_register_benchmark(TOFUEvaluator)
```

### Add to configs  
Evaluator config files are in [`configs/eval`](../configs/eval/), e.g [`configs/eval/tofu.yaml`](../configs/eval/tofu.yaml).

Example: TOFU evaluator config file ([`configs/eval/tofu.yaml`](../configs/eval/tofu.yaml))

```yaml
# @package eval.tofu
defaults: # include all the metrics that come under the TOFU evaluator
  - tofu_metrics: # When you import a metric here, its configuration automatically populates the 
  # metrics mapping below, enabled by the @package directive at the top of each metric config file.
    - forget_quality
    - forget_Q_A_Prob
    - forget_Q_A_ROUGE
    - model_utility # populated in the metrics key as metrics.model_utility

handler: TOFUEvaluator
metrics: {} # lists a mapping from each evaluation metric listed above to its config 
output_dir: ${paths.output_dir} # set to default eval directory
forget_split: forget10
```

---

## Model  

To add a new model:  

### Implement and register a handler  
For all the models currently supported, HuggingFace's `AutoModelForCausalLM` and `AutoTokenizer` are used, and therefore the user doesn't need to add or register any handler.

__Note__: Currently, we do not support loading models modified with LoRA and related variants. If you wish use such features, please create define and register model handlers for this logic in [`src/model`](../src/model) and provide the config info as discussed next.

### Add to configs  
Model configurations contain details required to load the model+tokenizer such as paths, chat templating arguments, LoRA parameters etc. in [`configs/models`](../configs/models/).

Example: LLaMA-3.1 model config in [`configs/model/Llama-3.1-8B-Instruct.yaml`](../configs/model/Llama-3.1-8B-Instruct.yaml).

```yaml
model_args:
  pretrained_model_name_or_path: "meta-llama/Llama-3.1-8B-Instruct"
  attn_implementation: 'flash_attention_2'
  torch_dtype: bfloat16
tokenizer_args:
  pretrained_model_name_or_path: "meta-llama/Llama-3.1-8B-Instruct"
template_args:
  apply_chat_template: True
  system_prompt: You are a helpful assistant.
```

---

## Collator  

Different dataset formats might have different data collation logic to pad and organize sequences in a batch. We do not expect most users to require new collators, but we provide the option to extend this component if needed.  

### Implement a handler  
Collators implementing batch collation are implemented in [`src/collators`](../src/collators/), imported in [`src/collators/__init__.py`](../src/collators/__init__.py).

```python
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    def __init__(self, tokenizer, padding_side, index):
      ...
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
      ...
```

### Register Collator handler  
Register the collator to link the class to the configs via the class name in [`COLLATOR_REGISTRY`](../src/collators/__init__.py).

Example: Registering `DataCollatorForSupervisedDataset` 

```python
from collators.base import DataCollatorForSupervisedDataset
_register_collator(DataCollatorForSupervisedDataset)
```

### Add to configs  
Collator configurations are in [`configs/collator`](../configs/collator/).

```yaml
DataCollatorForSupervisedDataset:
  handler: DataCollatorForSupervisedDataset
  args:
    padding_side: right
```

---

## Experiment  

Adding an experiment config with default config details reduces the need to manually set and override the many components and attributes. There is no handler or registration required here, as this is done purely through Hydra.

These configs are found in [`configs/experiment`](../configs/experiment/).

More details on how to run and organise experiments are in [`docs/experiment.md`](experiment.md).

### Add to configs  
Experiment configurations specify the model, dataset, trainer, and evaluation components.
Example: a TOFU unlearning experiment configuration (from [`configs/experiment/unlearn/tofu/llama2.yaml`](../configs/experiment/unlearn/tofu/llama2.yaml)) involves setting the model, the trainer, the dataset, the evaluation benchmark and the various attributes involves in them.

```yaml
# @package _global_

defaults:
  - override /model: Llama-2-7b-chat-hf
  - override /trainer: GradAscent
  - override /data: unlearn
  - override /data/datasets@data.forget: TOFU_QA_forget
  - override /data/datasets@data.retain: TOFU_QA_retain
  - override /eval: tofu

# Now, modify and set specific attributes to configs imported above

# define variables here to populate multiple fields
forget_split: forget10 
retain_split: retain90
retain_logs_path: null

eval:
  tofu:
    forget_split: ${forget_split}
    retain_logs_path: ${retain_logs_path}
    
data:
  anchor: forget
  forget:
    TOFU_QA_forget: 
      args:
        hf_args:
          name: ${forget_split}
  retain:
    TOFU_QA_retain:
      args:
        hf_args:
          name: ${retain_split}

trainer:
  args:
    warmup_epochs: 1.0 # custom parameter
    learning_rate: 2e-5
    weight_decay: 0.01
    num_train_epochs: 10
    # save_strategy: steps
    # save_steps: 0.5

override task_name: llama2_unlearn
```