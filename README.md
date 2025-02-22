<div align="center">    

![*Open*Unlearning](assets/banner_dark.png#gh-dark-mode-only)
![*Open*Unlearning](assets/banner_light.png#gh-light-mode-only)
<h3><strong>An easily extensible framework unifying LLM unlearning evaluation benchmarks.</strong></h3>
<!-- [![Paper]()]() -->
<!-- [![Conference](url)]() -->
<!-- ![CI testing](url) -->
</div>

---

## 📖 Overview

We provide efficient and streamlined implementations of the TOFU, MUSE unlearning benchmarks while supporting 5 unlearning methods, 3+ datasets, 6+ evaluation metrics, and 7+ LLMs. Each of these can be easily extended to incorporate more variants.

We invite the LLM unlearning community to collaborate by adding their new unlearning methods, datasets and evaluation metrics here to expand OpenUnlearning's features and get feedback from wider usage.

## 🗃️ Available Components

We provide several variants for each of the components in the unlearning pipeline.

| **Component**          | **Available Options** |
|----------------------|----------------------|
| **Benchmarks**       | [TOFU](https://arxiv.org/abs/2401.06121), [MUSE](https://muse-bench.github.io/) |
| **Unlearning Methods** | GradAscent, GradDiff, NPO, SimNPO, DPO |
| **Evaluation Metrics** | Verbatim Probability, Verbatim ROUGE, QA-Rouge, MIA Attacks, TruthRatio, Model Utility |
| **Datasets**         | MUSE (News, Books), TOFU (forget01, forget05, forget10) |
| **Model families**   | LLaMA-2, LLaMA 3.1, LLaMA 3.2, Phi-1.5, Phi-3.5, Gemma |

---

## 📌 Table of Contents
- [📖 Overview](#-overview)
- [🗃️ Available Components](#-available)
- [⚙️ Engineering Features](#-features)
- [⚡ Quickstart](#-quickstart)
  - [🛠️ Environment Setup](#-environment-setup)
  - [📜 Running Baseline Experiments](#-baselines)
- [🧪 Running Experiments](#-experiments)
  - [🚀 Perform Unlearning](#-run-unlearning)
  - [📊 Perform an Evaluation](#-run-evaluation)
- [➕ How to Add New Components](#-how-to-add)
- [🔗 Support & Contributors](#-support)
- [Citation](#-citation)

---

## ⚡ Quickstart

We provide detailed scripts to run multiple baselines for unlearning and evaluation in the [`scripts`](/scripts/) directory.

### 🛠️ Environment Setup

```bash
conda create -n unlearning python=3.11
conda activate unlearning
pip install -r requirements.txt
pip install flash-attn==2.6.3 --no-build-isolation
```

### 📜 Running Baseline Experiments
The scripts below execute standard baseline unlearning experiments on the TOFU and MUSE datasets, evaluated using their corresponding benchmarks.

```bash
bash scripts/tofu_unlearn.sh
bash scripts/muse_unlearn.sh
```

---

## 🧪 Running Experiments

We provide an easily configurable setup for running evaluations by leveraging Hydra configs. For a more detailed documentation of running experiments, including distributed training and simple finetuning of models, refer [`docs/experiments.md`](docs/experiments.md).

<!-- --- -->

### 🚀 Perform Unlearning

An example command for launching an unlearning process with `GradAscent` on the MUSE News dataset:

```bash
python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/llama2 \
  data_split=News trainer=GradAscent \
  trainer.args.num_train_epochs=10 
```

- `experiment`: Path to the Hydra config file [`configs/experiment/unlearn/muse/llama2.yaml`](configs/experiment/unlearn/muse/llama2.yaml) with default experimental settings for LLaMA 2 MUSE unlearning.
- `data_split`: Overrides the dataset split to use the MUSE News dataset.
- `trainer`: Overrides the unlearning algorithm using the Trainer defined in [`src/trainer/unlearn/grad_ascent.py`](src/trainer/unlearn/grad_ascent.py). `trainer.args.num_train_epochs=10` overrides a specific training argument.

### 📊 Perform an Evaluation

An example command for launching a TOFU evaluation process:

```bash
python src/eval.py --config-name=eval.yaml experiment=eval/tofu/llama2 \
  model.model_args.pretrained_model_name_or_path=<LOCAL_MODEL_PATH>
```

- `experiment`: Path to the evaluation configuration [`configs/experiment/eval/tofu/llama2.yaml`](configs/experiment/eval/tofu/llama2.yaml).
- `model.model_args.pretrained_model_name_or_path`: Overrides the default experiment config to evaluate a model from a local path.

For more details about running evaluations, refer [`docs/evaluation.md`](docs/evaluation.md).

---

## ➕ How to Add New Components

Adding a new component (trainer, evaluation metric, benchmark, model, or dataset) requires defining a new class, registering it, and creating a configuration file. Learn more about adding new components in [`docs/components.md`](docs/components.md).

Please feel free to raise a pull request with any features you add!

---

## 🔗 Support & Contributors

Developed and maintained by Vineeth Dorna ([@Dornavineeth](https://github.com/Dornavineeth)) and Anmol Mekala ([@molereddy](https://github.com/molereddy)) .

If you encounter any issues or have questions, feel free to raise an issue in the repository 🛠️.

## Citation

This repo is inspired from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We acknowledge the [TOFU](https://github.com/locuslab/tofu) and [MUSE](https://github.com/jaechan-repo/muse_bench) benchmarks, which served as the foundation for our re-implementation.

---

If you use OpenUnlearning in your research, please cite:

```bibtex
@misc{openunlearning2024,
  title={OpenUnlearning: A Unified Framework for LLM Unlearning Benchmarks},
  author={Dorna, Vineeth and Mekala, Anmol and Maini, Pratyush},
  year={2024},
  note={\url{https://github.com/Dornavineeth/OpenUnlearning}}
}
@inproceedings{maini2024tofu,
  title={TOFU: A Task of Fictitious Unlearning for LLMs},
  author={Maini, Pratyush and Feng, Zhili and Schwarzschild, Avi and Lipton, Zachary Chase and Kolter, J Zico},
  booktitle={First Conference on Language Modeling},
  year={2024}
}
```

---

## 📄 License
This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.