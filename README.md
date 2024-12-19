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
| [MUSE](https://muse-bench.github.io/)           |   ✅     |


## Tasks at hand

For detailed documentation please find the links below.
| Tasks | Link                                            |
|-----------|------------------------------------------------|
| Implement methods and run unlearning benchmarks      | [here](docs/unlearning.md)     |
| Implement and run evaluations for unlearning  | [here](docs/evaluation.md)     |
| Finetune models to learn datasets             | [here](docs/finetune.md)       |


## Quick Start

We provide detailed scripts to run multiple baselines for unlearning and their evaluations in the [scripts](/scripts/) directory.

### Unlearning in TOFU

Example script for launching an unlearning process with `GradDiff`.

```script
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/llama2 \
forget_split=forget10 \
trainer=GradDiff \
task_name=llama2_forget10_GradDiff \
retain_logs_path=saves/eval/tofu_forget10_retain/TOFU_EVAL.json
```
- **retain_logs_path**: For metrics such as forget quality metrics, which are tracked during unlearning, depend on the retained model. To compute these metrics, evaluation must be conducted using the retained model first.

All the parameters including training paramters, data splits, evaluation metrics can be found in [configs/experiment/unlearn/tofu/llama2](configs/experiment/unlearn/tofu/llama2.yaml)

### Evaluation in TOFU

Evaluate unlearned models using TOFU benchmark.

```script
python src/eval.py experiment=eval/tofu/llama2.yaml \
model.model_args.pretrained_model_name_or_path=locuslab/tofu_ft_llama2-7b \
forget_split=forget10 \
task_name=tofu_forget10_target \
retain_logs_path=saves/eval_reference/tofu_forget10/TOFU_EVAL.json
```

- **model.model_args.pretrained_model_name_or_path=locuslab/tofu_ft_llama2-7b**: Specifies the model to evaluate. This can be set to the unlearned model when evaluation of the unlearned model is required.

All the parameters including data splits, evaluation metrics can be found in [configs/experiment/eval/tofu/llama2](configs/experiment/eval/tofu/llama2.yaml)

### Unlearning in MUSE

```script
python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/llama2 \
data_split=News \
trainer=GradDiff \
task_name=llama2_news_GradDiff \
retain_logs_path=saves/eval/muse_news_retain/MUSE_EVAL.json
```
- **retain_logs_path**: For metrics such as privleak metrics, which are tracked during unlearning, depend on the retained model. To compute these metrics, evaluation must be conducted using the retained model first.

All the parameters including, data splits, evaluation metrics can be found in [configs/experiment/unlearn/muse/llama2](configs/experiment/unlearn/muse/llama2.yaml)


### Evaluation in MUSE

```script
python src/eval.py experiment=eval/muse/llama2.yaml \
model.model_args.pretrained_model_name_or_path=muse-bench/MUSE-News_target \
data_split=News \
task_name=muse_news_target \
retain_logs_path=saves/eval/muse_news_retain/MUSE_EVAL.json
```
- **model.model_args.pretrained_model_name_or_path=muse-bench/MUSE-News_target**: Specifies the model to evaluate. This can be set to the unlearned model when evaluation of the unlearned model is required.

All the parameters including, data splits, evaluation metrics can be found in [configs/experiment/eval/muse/llama2](configs/experiment/eval/muse/llama2.yaml)


### Train models for developing new benchmark

Run finetuning of TOFU dataset with the following script

```bash
python src/train.py experiment=finetune/tofu/llama2_inst_full
trainer.args.learning_rate=5e-5
```
You can set or override all training parameters via the command line by passing them in `trainer.args`, as demonstrated with `learning_rate` above.

All the parameters including, data splits, evaluation metrics can be found in [configs/experiment/finetune/tofu/llama2_inst_full](configs/experiment/finetune/tofu/llama2_inst_full)



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