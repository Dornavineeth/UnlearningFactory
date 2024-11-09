<div align="center">    
 
# Unlearning Factory    

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://arxiv.org/abs/2401.06121)
[![Conference](http://img.shields.io/badge/COLM-2024-4b44ce.svg)](https://openreview.net/forum?id=B41hNBoWLo)
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Overview
A unified repository for developing benchmarks for unlearning. This repository is designed to accelerate research by streamlining your workflow in unlearning. The key features include:

- Unlearning Models: Effortlessly implement unlearning techniques for your LLM.
- Evaluation of Unlearned Models: Assess the performance of models post-unlearning with comprehensive and custom evaluation metrics.
- Fine-tuning Models: Fine-tune your models to develop new bechmark.


## Installation
```bash
# Pip installation
conda activate unlearning
pip install -r requirements.txt


# Conda Installation
conda create unlearning python=3.11
conda activate unlearning
pip install -r requirements.txt
```
## Benchmarks Included

| Benchmark | Support                                        |
|-----------|------------------------------------------------|
| [TOFU](https://arxiv.org/abs/2401.06121)        | ✅       |
| [MUSE](https://muse-bench.github.io/)           |        |


## Tasks at hand

For detailed documentation please find the links below.
| Tasks | Link                                            |
|-----------|------------------------------------------------|
| Unlearn and implement new unlearning methods       | [here](docs/unlearning.md)     |
| Evaluate and implement new metrics for unlearning  | [here](docs/evaluation.md)     |
| Train models for developing new benchmark.              | [here](docs/finetune.md)       |


## Quick Start

### Unlearn and implement new unlearning methods 

Example script for launching an unlearning process with `GradDiff`.

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


## Evaluate and implement new metrics for unlearning

To run TOFU benchmark
```bash
python src/eval.py \
model=Llama-3.1-8B-Instruct \ # Model to evaluate
eval=tofu # evaluation benchmark to run
output_dir=evals # set the output directory to store results
```

- **model=Llama-3.1-8B-Instruct**: Loads the model configuration from [Llama-3.1-8B-Instruct.yaml](../configs/model/Llama-3.1-8B-Instruct.yaml)
- **trainer=finetune**: Loads the [Hugging Face's](https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py) `Trainer` with the configuration defined in [finetune.yaml](../configs/trainer/finetune.yaml).
- **eval=tofu**: Specifies the [tofu.yaml](../configs/eval/tofu.yaml) config for [TOFU](https://arxiv.org/abs/2401.06121) benchmark evaluation.
- **output_dir=evals**: Specifies the output directory for storing results.


## Train models for developing new benchmark

Quickly launch finetuning job with the following script

```bash
python src/train.py
model=Llama-3.1-8B-Instruct \
trainer=finetune \
data.train=TOFU_QA_FULL \
data.eval=TOFU_QA_FORGET10_P \
collator=DataCollatorForSupervisedDataset
```

- **model=Llama-3.1-8B-Instruct**: Loads the model configuration from [Llama-3.1-8B-Instruct.yaml](../configs/model/Llama-3.1-8B-Instruct.yaml)
- **trainer=finetune**: Loads the [Hugging Face's](https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py) `Trainer` with the configuration defined in [finetune.yaml](../configs/trainer/finetune.yaml).
- **trainer.method_args.alpha=0.5**: Overrides the alpha parameter [GradDiff.yaml](../configs/trainer/GradDiff.yaml).
- **data.train=TOFU_QA_FULL**: Sets the train dataset to load [QADataset](../src/data/tofu.py) with config [TOFU_QA_FULL.yaml](../configs/data/datasets/TOFU_QA_FULL.yaml).
- **data.eval=TOFU_QA_FORGET10_P**: Sets the evaluation dataset for [QADataset](../src/data/tofu.py) with config [TOFU_QA_FORGET10_P.yaml](../configs/data/datasets/TOFU_QA_FORGET10_P.yaml).


<!-- ##
## 
### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```    -->