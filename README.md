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

| Tasks | Link                                            |
|-----------|------------------------------------------------|
| Unlearn and implement new unlearning methods       | [here](docs/unlearning.md)     |
| Evaluate and implement new metrics for unlearning  | [here](docs/evaluation.md)     |
| Train models for developing new benchmark.              | [here](docs/finetune.md)       |



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