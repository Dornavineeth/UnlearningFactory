<div align="center">    
 
# Evaluation

</div>

Our framework supports the evaluation of multiple benchmarks specifically for unlearning tasks.



## Sample Evaluation Script

```bash
python src/eval.py \
model=Llama-3.1-8B-Instruct \ # Model to evaluate
eval=tofu # evaluation benchmark to run
```

- **model=Llama-3.1-8B-Instruct**: Loads the model configuration from [Llama-3.1-8B-Instruct.yaml](../configs/model/Llama-3.1-8B-Instruct.yaml)
- **trainer=finetune**: Loads the [Hugging Face's](https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py) `Trainer` with the configuration defined in [finetune.yaml](../configs/trainer/finetune.yaml).
- **eval=tofu**: Specifies the [tofu.yaml](../configs/eval/tofu.yaml) config for [TOFU](https://arxiv.org/abs/2401.06121) benchmark evaluation.


## Evaluation Benchmark Config

Each benchmark configuration includes a list of metrics to be evaluated.

Sample config [tofu.yaml](../configs/eval/tofu.yaml) of [TOFU](https://arxiv.org/abs/2401.06121) benchmark.
```yaml
# @package eval.tofu

defaults: # include all defined metrics files
  - tofu_metrics: 
    - Q_A_Prob
    - Q_PARA_A_PARA_Prob
    - Q_A_ROUGE
    - Q_A_PARA_ROUGE


device: cuda # device to load model
output_dir: ${output_dir} # set to default eval directory
metrics: {}
```

- **tofu_metrics**: Lists all the metrics to be evaluated, where each metric configuration is sourced from [configs/eval/tofu_metrics](../configs/eval/tofu_metrics/)
- **output_dir**: Specifies the directory to save evaluation results.

__NOTE__: The first line `@package eval.tofu` is not a comment, but populates `eval` key in [eval.yaml](../configs/eval.yaml) with tofu. See [Hydra](https://hydra.cc/docs/upgrades/0.11_to_1.0/adding_a_package_directive/) documenation.


## Metric Configuration


Each metric has its own configuration, which specifies the datasets, collators, and any additional arguments needed for calculation.

Sample Config [Q_A_Prob.yaml](../configs/eval/tofu_metrics/Q_A_Prob.yaml) for a metric `Q_A_Prob` which calculates probability of Question Answer pairs.
```yaml
# @package eval.tofu.metrics.Q_A_Prob
defaults:
  - ../../data/datasets@datasets: TOFU_QA_FORGET10
  - ../../collator@collators: DataCollatorForSupervisedDatasetwithIndex
  # ^ get default dataset and generation config information

batch_size: 16
```

- **../../data/datasets@datasets: TOFU_QA_FORGET10**: Loads the datasets configs [TOFU_QA_FORGET10](../configs/data/datasets/TOFU_QA_FORGET10.yaml) for the calculating probability. 
- **../../collator@collators**: Loads collator config [DataCollatorForSupervisedDatasetwithIndex](../configs/collator/DataCollatorForSupervisedDatasetwithIndex.yaml) for the calculating probability.

__NOTE__: The first line `@package eval.tofu.metrics.Q_A_Prob` is not a comment, but populates the `metrics` key in [eval/tofu.yaml](../configs/eval/tofu.yaml). See [Hydra](https://hydra.cc/docs/upgrades/0.11_to_1.0/adding_a_package_directive/) documenation.

## Metric Implementation


A metric is implemented as a function decorated with `@unlearning_metric`. This decorator wraps the function, automatically loading and processing the datasets and collators specified in the configuration, so they are readily available for use in the function.

Implementation of `Q_A_Prob` Metric

```python
# src/evals/metrics/memorization.py

@unlearning_metric(name="Q_A_Prob")
def q_a_prob(model, **kwargs):
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]

    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
    index_to_probs = evaluate_probability(model, dataloader)
    return index_to_probs
```

- **@unlearning_metric(name="Q_A_Prob")**: Defines a metric with name as `Q_A_Prob`.
- `datasets` and `collators` defined in config [Q_A_Prob.yaml](../configs/eval/tofu_metrics/Q_A_Prob.yaml) are automatically loaded with their data handlers and can be accessed in kwargs.


## Register Metric

Link the metric configuration to the unlearning_metric function implemented in [__init__.py](../src/evals/metrics/__init__.py).

Example to register  `Q_A_Prob` Metric

```python
from evals.metrics.memorization import q_a_prob

_register_metric(q_a_prob)
```
