<div align="center">    
 
# Finetune Models

</div>



## Preliminaries
Refer to the [Components Guide](/docs/components.md) for instructions on adding individual components such as datasets, models, and trainers.



## Finetune

Quickly run finetuning with the following script


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