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
pip install -r requirements.txt


# Conda Installation
conda create -n unlearning python=3.11
conda activate unlearning
pip install -r requirements.txt
pip install flash-attn==2.6.3 --no-build-isolation
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
| Implement methods and run unlearning       | [here](docs/unlearning.md)     |
| Implement and run evaluations for unlearning  | [here](docs/evaluation.md)     |
| Finetune models to learn datasets             | [here](docs/finetune.md)       |


## Quick Start

### Implement methods and run unlearning 

Example script for launching an unlearning process with `GradDiff`.

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
- **model.model_args.pretrained_model_name_or_path=LOCAL PATH**: Overrides the model path defined in [Llama-3.1-8B-Instruct.yaml](../configs/model/Llama-3.1-8B-Instruct.yaml) to provide path to pre-unlearning finetuned model.
- **trainer=GradDiff**: Loads the unlearning trainer [GradDiffTrainer](../src/trainer/unlearn/grad_diff.py) with the configuration defined in [GradDiff.yaml](../configs/trainer/GradDiff.yaml).
- **trainer.method_args.alpha=0.5**: Overrides the alpha parameter [GradDiff.yaml](../configs/trainer/GradDiff.yaml).
- **data.forget=TOFU_QA_forget**: Sets the forget dataset to load [QADataset](../src/data/tofu.py) with config [TOFU_QA_forget.yaml](../configs/data/datasets/TOFU_QA_forget.yaml) for unlearning.
- **data.retain=TOFU_QA_retain**: Sets the retain dataset for [QADataset](../src/data/tofu.py) with config [TOFU_QA_retain.yaml](../configs/data/datasets/TOFU_QA_retain.yaml) for unlearning.
- **data.eval=TOFU_QA_forget_para**: Sets the evaluation dataset for [QADataset](../src/data/tofu.py) with config [TOFU_QA_forget_para.yaml](../configs/data/datasets/TOFU_QA_forget_para.yaml) for unlearning.


### Evaluate and implement new metrics for unlearning

To run TOFU benchmark
```bash
python src/eval.py \
# Model to evaluate
model=Llama-3.1-8B-Instruct \ 
# Evaluation config to run (e.g. tofu benchmark)
eval=tofu \ 
# Set the output directory to store results
output_dir=evals 
```

- **model=Llama-3.1-8B-Instruct**: Loads the model configuration from [Llama-3.1-8B-Instruct.yaml](../configs/model/Llama-3.1-8B-Instruct.yaml)
- **trainer=finetune**: Loads the [Hugging Face's](https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py) `Trainer` with the configuration defined in [finetune.yaml](../configs/trainer/finetune.yaml).
- **eval=tofu**: Specifies the [tofu.yaml](../configs/eval/tofu.yaml) config for [TOFU](https://arxiv.org/abs/2401.06121) benchmark evaluation.
- **output_dir=evals**: Specifies the output directory for storing results.


### Train models for developing new benchmark

Quick start: run finetuning with the following script

```bash
python src/train.py \
model=Llama-3.1-8B-Instruct \
trainer=finetune \
data.train=TOFU_QA_full \
data.eval=TOFU_QA_forget_para \
collator=DataCollatorForSupervisedDataset
```

- **model=Llama-3.1-8B-Instruct**: Loads the model configuration from [Llama-3.1-8B-Instruct.yaml](../configs/model/Llama-3.1-8B-Instruct.yaml)
- **trainer=finetune**: Loads the [Hugging Face's](https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py) `Trainer` with the configuration defined in [finetune.yaml](../configs/trainer/finetune.yaml).
- **trainer.method_args.alpha=0.5**: Overrides the alpha parameter [GradDiff.yaml](../configs/trainer/GradDiff.yaml).
- **data.train=TOFU_QA_full**: Sets the train dataset to load [QADataset](../src/data/tofu.py) with config [TOFU_QA_full.yaml](../configs/data/datasets/TOFU_QA_full.yaml).
- **data.eval=TOFU_QA_forget_para**: Sets the evaluation dataset for [QADataset](../src/data/tofu.py) with config [TOFU_QA_forget_para.yaml](../configs/data/datasets/TOFU_QA_forget_para.yaml).


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