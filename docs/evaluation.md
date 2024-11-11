<div align="center">    
 
# Evaluation

</div>

UnlearningFactory supports the evaluation of multiple unlearning benchmarks.



## Quick Start
Run the TOFU benchmark evaluations:
```bash
python src/eval.py \
# Model to evaluate
model=Llama-3.1-8B-Instruct \ 
# Evaluation benchmark to run
eval=tofu \ 
# Set the output directory to store results
output_dir=evals 
```

- **model=Llama-3.1-8B-Instruct**: Loads the model configuration from [Llama-3.1-8B-Instruct.yaml](../configs/model/Llama-3.1-8B-Instruct.yaml)
- **trainer=finetune**: Loads the [Hugging Face's](https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py) `Trainer` with the configuration defined in [finetune.yaml](../configs/trainer/finetune.yaml).
- **eval=tofu**: Specifies the [tofu.yaml](../configs/eval/tofu.yaml) config for [TOFU](https://arxiv.org/abs/2401.06121) benchmark evaluation.
- **output_dir=evals**: Specifies the output directory for storing results.


## Evaluation Benchmark Config

Each benchmark configuration includes a list of metrics to be evaluated.

Sample config [tofu.yaml](../configs/eval/tofu.yaml) of [TOFU](https://arxiv.org/abs/2401.06121) benchmark.
```yaml
# @package eval.tofu

defaults: # include all defined metrics files
  - tofu_metrics: 
    - Q_A_Prob # Probability on QA
    - Q_PARA_A_PARA_Prob # Probability on paraphrased question and paraphrased anser
    - Q_A_ROUGE # ROUGE on QA
    - Q_A_PARA_ROUGE # ROUGE on question and paraphrased answer

device: cuda # device to load model
output_dir: ${output_dir} # set to default eval directory
metrics: {} # Will be populated by tofu_metrics
```

- **tofu_metrics**: Lists all the metrics to be evaluated, where each metric configuration is sourced from [configs/eval/tofu_metrics](../configs/eval/tofu_metrics/)
- **output_dir**: Specifies the directory to save evaluation results for this particular benchmark.

__NOTE__: The first line `@package eval.tofu` is not a comment, but populates `eval` key in [eval.yaml](../configs/eval.yaml) with tofu. See [Hydra](https://hydra.cc/docs/upgrades/0.11_to_1.0/adding_a_package_directive/) documenation.


## Metric Configuration


Each metric has its own configuration, which specifies the datasets, collators, and any additional arguments needed for calculation.

Sample Config [Q_A_Prob.yaml](../configs/eval/tofu_metrics/Q_A_Prob.yaml) for a metric with name `Q_A_Prob` which calculates probability of Question Answer pairs.
```yaml
# @package eval.tofu.metrics.Q_A_Prob

defaults:
  - ../../data/datasets@datasets: TOFU_QA_FORGET10
  - ../../collator@collators: DataCollatorForSupervisedDatasetwithIndex
  # ^ get default dataset and generation config information

handler: probability
batch_size: 16
```

- **../../data/datasets@datasets: TOFU_QA_FORGET10**: Loads the datasets configs [TOFU_QA_FORGET10](../configs/data/datasets/TOFU_QA_FORGET10.yaml) for the calculating probability. 
- **../../collator@collators**: Loads collator config [DataCollatorForSupervisedDatasetwithIndex](../configs/collator/DataCollatorForSupervisedDatasetwithIndex.yaml) for the calculating probability.
- **handler: probability**: Specifies the core implemented unlearning metric handler to use for evaluation.


__NOTE__: 
- The prefix `../../data/datasets` and `../../collator` are used to locate the `datasets` amd `collator` config packages w.r.t [tofu_metrics](../configs/eval/tofu_metrics/) sub-package,
- The first line `@package eval.tofu.metrics.Q_A_Prob` is not a comment, but populates the `metrics` key in [eval/tofu.yaml](../configs/eval/tofu.yaml). See [Hydra](https://hydra.cc/docs/upgrades/0.11_to_1.0/adding_a_package_directive/) documenation.

## Metric Handler Implementation




A metric handler is implemented as a function decorated with `@unlearning_metric`. This decorator wraps the function into an UnlearningMetric object. This helps to automatically load and prepare datasets and collators for `probability` as specified in the eval config ([example](../configs/eval/tofu_metrics/Q_A_Prob.yaml)), so they are readily available for use in the function.

Implementation of `probability` Metric

```python
# src/evals/metrics/memorization.py

@unlearning_metric(name="probability")
def probability(model, **kwargs):
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]

    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
    index_to_probs = evaluate_probability(model, dataloader)
    return index_to_probs
```

- **@unlearning_metric(name="probability")**: Defines a metric with handler name as `probability`.
- `datasets` and `collators` defined in config [Q_A_Prob.yaml](../configs/eval/tofu_metrics/Q_A_Prob.yaml) are automatically loaded with their data handlers and can be accessed in kwargs.


## Register Metric

Register the `unlearning_metric` function implemented in [__init__.py](../src/evals/metrics/__init__.py).

Example to register  `probability` Metric

```python
from evals.metrics.memorization import probability

_register_metric(probability)
```

## Advanced Features

### Pre-Compute

In TOFU, we support the creation of aggregated metrics that depend on other metrics computed previously. For example, the `truth_ratio` metric relies on both *perturbed* and *paraphrased* probabilities for questions and answers, so its implementation could utilize existing metric and their evaluations. To avoid redundant computation, we provide a pre-compute option, allowing metrics to be calculated in advance and referenced in other metrics. Then while, implementing the dependent metric, the evals for the others can be accessed and used directly.

```yaml
# @package eval.tofu.metrics.truth_ratio

defaults:
  - .@pre_compute.Q_A_PARA_Prob: Q_A_PARA_Prob
  - .@pre_compute.Q_A_PERT_Prob: Q_A_PERT_Prob

pre_compute:
  Q_A_PARA_Prob:
    access_key: paraphrase
  Q_A_PERT_Prob:
    access_key: perturb

handler: truth_ratio
```
- **.@pre_compute.Q_A_PARA_Prob: Q_A_PARA_Prob**: This specifies that, for the current package (`.`), we are renaming a configuration to pre_compute with a key of `Q_A_PARA_Prob`. The configuration parameters for this key are loaded from the file [Q_A_PARA_Prob.yaml](../configs/eval/tofu_metrics/Q_A_PARA_Prob.yaml).
- **handler: truth_ratio**: Specifies the handler implemented for the aggregated metric evaluation.
  
```yaml
pre_compute:
  Q_A_PARA_Prob:
    access_key: paraphrase
```
- **access_key**: Specifies how the pre-computed results should be accessed within the `truth_ratio` handler. Here, the `Q_A_PARA_Prob` results are accessible under the key `paraphrase`, while `Q_A_PERT_Prob` is accessible under `perturb`.

```python
@unlearning_metric(name="truth_ratio")
def truth_ratio(model, **kwargs):
    para_results = kwargs["pre_compute"]["paraphrase"]
    pert_results = kwargs["pre_compute"]["perturb"]
    index_to_scores = {}
    for k, para_result in para_results.items():
        para_prob = para_result["prob"]
        pert_result = pert_results[k]
        pert_prob = sum([r["prob"] for r in pert_result]) / len(pert_result)
        index_to_scores[k] = {"truth_ratio": pert_prob / para_prob}
    return index_to_scores
```
- Access the pre-computed metric results of paraphrased probability through `kwargs["pre_compute"]["paraphrase"]`.
