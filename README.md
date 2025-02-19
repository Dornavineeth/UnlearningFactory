<div align="center">    
 
# OpenUnlearning | <strong style="font-size:0.75em">easy LLM unlearning </strong>  

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://arxiv.org/abs/2401.06121)
[![Conference](http://img.shields.io/badge/COLM-2024-4b44ce.svg)](https://openreview.net/forum?id=B41hNBoWLo) -->

<!-- ARXIV    -->
<!-- [![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539) -->

<!-- ![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push) -->
 
<!-- Conference -->
   
</div>
 
## Overview
A framework unifying LLM unlearning benchmarks.

We provide efficient and streamlined implementations of the TOFU, MUSE unlearning benchmarks while supporting 5 unlearning methods, 3+ datasets, 6+ evaluation metrics, and 7+ LLMs. Each of these can be easily extended to incorporate more variants.

### Supported Components

| **Component**          | **Available Options** |
|----------------------|----------------------|
| **Benchmarks**       | [TOFU](https://arxiv.org/abs/2401.06121), [MUSE](https://muse-bench.github.io/) |
| **Unlearning Methods** | GradAscent, GradDiff, NPO, SimNPO, DPO |
| **Evaluation Metrics** | Verbatim Probability, Verbatim ROUGE, QA-Rouge, MIA Attacks, TruthRatio, Model Utility |
| **Datasets**         | MUSE (News, Books), TOFU (forget01, forget05, forget10) |
| **LLMs**             | LLaMA-2, LLaMA 3.1, LLaMA 3.2, Phi-1.5, Phi-3.5, Gemma |

## Engineering Features

- **Multi-GPU Training**: Supported via DeepSpeed and Accelerate.  
- **Extensibility**: Allows for easy addition and evaluation of new unlearning methods, datasets, and benchmark tasks.  
- **Experiment Management**: Uses Hydra for streamlined experiment configuration. 
<!-- (can we mention hp tuning?).   -->
- **Efficient Evaluation**: Supports batched evaluation and metric aggregation for streamlined tracking.  

## Quickstart

We provide detailed scripts to run multiple baselines for unlearning and evaluation in the [`scripts`](/scripts/) directory.

Environment setup

```bash
pip install -r requirements.txt

conda create -n unlearning python=3.11
conda activate unlearning
pip install -r requirements.txt
pip install flash-attn==2.6.3 --no-build-isolation
```


Running standard baseline experiments on TOFU and MUSE

```bash
bash scripts/tofu_unlearn.sh
bash scripts/muse_unlearn.sh
```
</details>

### Running an experiment

**Note:** We defined some default experimental configs in the [`configs/experiment`](configs/experiment) directory and used them below.

1. **Unlearning:** An example command for launching an unlearning process with `GradAscent` on the MUSE News dataset:

```bash
python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/llama2 \
  data_split=News trainer=GradAscent \
  trainer.args.num_train_epochs=10 
```
- `experiment`: path to the Hydra config file [`configs/experiment/unlearn/muse/llama2.yaml`](configs/experiment/unlearn/muse/llama2.yaml) with default experimental settings for Llama2 MUSE unlearning, which are used to populate the Hydra config for this experimental run. These can be overriden (see next).
- `data_split`: overrides the dataset split to use the MUSE News dataset. Check the [experiment config](configs/experiment/unlearn/muse/llama2.yaml) to see how this argument is used.
- `trainer`: overrides the unlearning algorithm to use the HuggingFace Trainer from [`src/trainer/unlearn/grad_ascent.py`](src/trainer/unlearn/grad_ascent.py). `trainer.args.num_train_epochs=10` overrides specific training arguments.

2. **Run evaluation**: A example command for launching a TOFU evaluation process on a saved model checkpoint.

```bash
python src/eval.py --config-name=eval.yaml experiment=eval/tofu/llama2 \
  model.model_args.pretrained_model_name_or_path=<LOCAL_MODEL_PATH>
```
- `experiment`: [`configs/experiment/eval/tofu/llama2.yaml`](configs/experiment/eval/tofu/llama2.yaml) is the experiment config.
- `model.model_args.pretrained_model_name_or_path`: overrides the default experiment config to evaluate a model from a specific local path.

3. **Run finetuning**

We provide for simple LLM finetuning on datasets, which can be used to prepare models for a benchmark before unlearning. For example, to train a model on the TOFU dataset:
```bash
python src/train.py --config-name=finetune.yaml experiment=finetune/tofu/llama2_inst_full
```

## Adding new components

Adding a new component, such as a trainer (for finetuning or unlearning), evaluation metric, benchmark or dataset requires adding a new class, registering it to a name and creating a config with the name. This procedure is documented in the links below.

| Tasks | Link |
|-----------|------------------------------------------------|
| General: How components are designed and how to add new ones | [`docs/components.md`](docs/components.md) |
| Adding unlearning methods | [`docs/unlearning.md`](docs/unlearning.md) |
| Adding unlearning evaluations | [`docs/evaluation.md`](docs/evaluation.md) |


### Acknowledgement
This repo is inspired from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) which unifies LLM finetuning. We also acknowledge the [TOFU](https://github.com/locuslab/tofu) and [MUSE](https://github.com/jaechan-repo/muse_bench) benchmarks, which served as the foundation for our re-implementation.

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