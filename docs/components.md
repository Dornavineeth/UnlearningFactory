# Components  

The OpenUnlearning framework requires a structured approach for adding new components in the unlearning pipeline.

This usually involves 3 steps:
1. __Implementing a handler__: This implements logic which the component uses. A single handler may be used by multiple components. For example, the handler for ROUGE score computation may support multiple evaluation metrics on various datasets.
2. __Registering the handler__: This collects all the created handlers of a kind into a single mapping so that they can be accessed by a key.
3. __Adding a config file__: Here we set up a component in a Hydra config that uses the previously defined handler and set configuration hyperparameters for usage. These config files can now be provided in the arguments to the python script.

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
# TODO generic process
---

## Trainer  

To add a new **Trainer**:  

### Implement a handler  
We extend HuggingFace's [`Trainer`](https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py) for our trainers. Trainer handlers implementing custom training algorithms are written in [`src/trainer`](../src/trainer/).  

Example: defining a gradient-difference based unlearning Trainer.

```python
class GradDiff(UnlearnTrainer):
    def __init__(self, gamma, alpha, ...):
        ...
      
    def compute_loss(self, model, inputs, return_outputs=False):
        ...
```

### Register Trainer handler  
Register the handler to link the class to the configs via the class name in [`TRAINER_REGISTRY`](../src/trainer/__init__.py).

Example: Registering a **fine-tuning trainer** and `GradDiff`, an **unlearning trainer**  

```python
from transformers import FinetuneTrainer
from trainer.unlearn.grad_ascent import GradDiff
_register_trainer(FinetuneTrainer) # class defined in src/trainer/base.py
_register_trainer(GradDiff) # class defined in src/trainer/unlearn/grad_diff.py
```

### Add a trainer to configs  

Add a config that uses the new trainer and set parameters. Trainer configurations are in [`configs/trainer`](../configs/trainer/). Each config contains a handler that points to the defined trainer class and the arguments used to initialise the trainer.

Example: add a config file ([`configs/trainer/GradDiff.yaml`](configs/trainer/GradDiff.yaml)) for the GradDiff approach using the defined `GradDiff` trainer handler.
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
We extend `torch.utils.data.Dataset` to implement Dataset handlers for loading and preprocessing data. These are written in [`src/data`](../src/data/). A new dataset is instantiated by providing its parameters (dataset column, length etc) to an existing dataset handler.

Example: defining a `PretrainingDataset` dataset handler to load long texts for pre-training style next token prediction.

```python
class PretrainingDataset(Dataset):
    def __init__(self, hf_args, text_key, max_length, ...):
        ...

    def __getitem__(self, idx):
        ...
        return item
```

### Register Dataset handler  
Register the handler to link the class to the configs via the class name in [`DATASET_REGISTRY`](../src/data/__init__.py).

Example: Registering `PretrainingDataset`  

```python
from data.pretraining import PretrainingDataset
_register_data(PretrainingDataset)
```

### Add a dataset to configs  
Add a specific dataset that uses the `PretrainingDataset` class format. Dataset configurations are in [`configs/data/datasets`](../configs/data/datasets/). Each config contains a handler that points to the defined dataset class and the arguments used to create the dataset.

Example: add a config file for the `MUSE_forget` dataset using the `PretrainingDataset` handler
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
```
# TODO add another dataset that uses same handler
---

## Evaluation Metric  

To add a new evaluation metric, we create a handler with the metric computation logic and then configure it.

### Implement a handler  
Metric handlers are implemented in [`src/evals/metrics`](../src/evals/metrics/), where we define handlers containing generic logic to compute individual statistics and/or aggregated metrics over a dataset like ROUGE scores, TOFU's Forget Quality etc.

Example: implementing `forget_quality` and `rouge` handlers

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

# in src/evals/metrics/privacy.py
@unlearning_metric(name="forget_quality")
def forget_quality(model, **kwargs): 
  # the forget quality metric is aggregated from computed statistics of 
  # other metrics like truth ratio, which is provided through kwargs
  ...
  return {"agg_value": pvalue}
```

### Register Metric handler  
Register the handler to link the class to the configs via the class name in [`METRIC_REGISTRY`](../src/evals/metrics/__init__.py).

Example: Registering `rouge` handler  

```python
from evals.metrics.memorization import rouge
_register_metric(rouge)
```

### Add a metric to configs  
Metric configurations are in [`configs/eval/tofu_metrics`](../configs/eval/tofu_metrics/) and [`configs/eval/muse_metrics`](../configs/eval/muse_metrics/). These create individual evaluation metrics by providing the handler a specific dataset and other parameters.

Example 1: Creating the config for MUSE's `forget_verbmem_ROUGE` metric (see [`configs/eval/muse_metrics/forget_knowmem_ROUGE.yaml`](configs/eval/muse_metrics/forget_knowmem_ROUGE.yaml)). 

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

Example 2: Creating the config for TOFU's `forget_quality` metric (see [`configs/eval/tofu_metrics/forget_quality.yaml`](configs/eval/tofu_metrics/forget_quality.yaml)). 

```yaml
# @package eval.tofu.metrics.forget_quality
defaults:
# since forget_quality depends on forget_Truth_Ratio, set a precompute to 
# enforce the computation of forget_Truth_Ratio before forget_quality
  - .@pre_compute.forget_truth_ratio: forget_Truth_Ratio

reference_logs:
  retain_model_logs:
    path: ${eval.tofu.retain_logs_path}
    include: 
      forget_truth_ratio:
        access_key: retain

pre_compute:
  forget_truth_ratio:
    access_key: forget

handler: forget_quality
```

---

## Benchmark  

A benchmark is a collection of evaluation metrics defined above (e.g. TOFU, MUSE). To add a new **Benchmark**:  

### Implement a handler  
Handlers for evaluating and aggregating a collection of metrics in a benchmark are implemented in [`src/evals`](../src/evals/), for example in [`src/evals/tofu.py`](../src/evals/tofu.py). These handlers will take the defined metrics listed in the config (see next) and provide for running the evaluation.

### Register Benchmark handler  
Register the benchmark to link the class to the configs via the class name in [`BENCHMARK_REGISTRY`](../src/evals/__init__.py).

Example: Registering TOFU benchmark  

```python
from evals.tofu import TOFUBenchmark
_register_benchmark(TOFUBenchmark)
```

### Add to configs  
Benchmark config files are in [`configs/eval`](../configs/eval/), e.g [`configs/eval/tofu.yaml`](configs/eval/tofu.yaml). Each config contains

```yaml
# @package eval.tofu
defaults: # include all the metrics that come under the TOFU evaluator
  - tofu_metrics: # When you import a metric here, its configuration automatically populates the 
  # metric key below, enabled by the @package directive at the top of each metric config file.
    - forget_quality
    - forget_Q_A_Prob
    - forget_Q_A_ROUGE
    - model_utility # populated in the metrics key as metrics.model_utility

handler: TOFUEvaluator
output_dir: ${paths.output_dir} # set to default eval directory
metrics: {} # lists a mapping from each evaluation metric to its config 
overwrite: false
forget_split: forget10
retain_logs_path: null
```

---

## Model  

To add a new **Model**:  

### Implement and regeister a handler  
For most cases, HuggingFace's `AutoModelForCausalLM` and `AutoTokenizer` are used, and the user doesn't need to add or register any handler.

### Add to configs  
Model configurations are in [`configs/models`](../configs/models/).

Example: LLaMA-3.1 model config in [`configs/model/Llama-3.1-8B-Instruct.yaml`](configs/model/Llama-3.1-8B-Instruct.yaml).

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

To add a new collator:  

### Implement a handler  
Collators handling batch collation are implemented in [`src/collators`](../src/collators/), imported in [`src/collators/__init__.py`](../src/collators/__init__.py).

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

To add a new **Experiment**:  

### Implement a handler  
Experiments combine model, dataset, trainer, and evaluation components. Each experiment is defined in [`configs/experiment`](../configs/experiment/).

### Add to configs  
Experiment configurations specify the model, dataset, trainer, and evaluation components.
Example: TOFU unlearning experiment  

```yaml
experiment: unlearn/tofu/llama2
trainer: GradAscent
model: llama2
dataset: TOFU_QA_full
eval: tofu
```