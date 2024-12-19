# TODO add each experiment in configs/experiment folder
python src/train.py --config-name=unlearn.yaml trainer=GradAscent
python src/train.py --config-name=unlearn.yaml trainer=GradDiff
python src/train.py --config-name=unlearn.yaml trainer=GradDiff trainer.method_args.retain_loss_type=KL
python src/train.py --config-name=unlearn.yaml trainer=NPO
python src/train.py --config-name=unlearn.yaml trainer=NPO trainer.method_args.retain_loss_type=KL
python src/train.py --config-name=unlearn.yaml trainer=DPO data/datasets@data.forget=TOFU_QA_forget_idk