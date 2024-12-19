########################################################################################################################
########################################### TOFU unlearning forget10 ###################################################
########################################################################################################################

python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/llama2 trainer=GradAscent task_name=llama2_forget10_GradAscent
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/llama2 trainer=GradDiff task_name=llama2_forget10_GradDiff
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/llama2 trainer=GradDiff trainer.method_args.retain_loss_type=KL task_name=llama2_forget10_GradDiff_KL
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/llama2 trainer=NPO task_name=llama2_forget10_NPO
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/llama2 trainer=NPO trainer.method_args.retain_loss_type=KL task_name=llama2_forget10_NPO_KL
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/llama2_idk trainer=DPO data/datasets@data.forget=TOFU_QA_forget_idk task_name=llama2_forget10_IdkDPO


########################################################################################################################
########################################### TOFU unlearning forget05 ###################################################
########################################################################################################################

# # TOFU unlearning forget05
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/llama2 trainer=GradAscent task_name=llama2_forget05_GradAscent forget_split=forget05 retain_split=retain95
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/llama2 trainer=GradDiff task_name=llama2_forget05_GradDiff forget_split=forget05 retain_split=retain95
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/llama2 trainer=GradDiff trainer.method_args.retain_loss_type=KL task_name=llama2_forget05_GradDiff_KL forget_split=forget05 retain_split=retain95
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/llama2 trainer=NPO task_name=llama2_forget05_NPO forget_split=forget05 retain_split=retain95
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/llama2 trainer=NPO trainer.method_args.retain_loss_type=KL task_name=llama2_forget05_NPO_KL forget_split=forget05 retain_split=retain95
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/llama2_idk trainer=DPO data/datasets@data.forget=TOFU_QA_forget_idk task_name=llama2_forget05_IdkDPO forget_split=forget05 retain_split=retain95


########################################################################################################################
########################################### TOFU unlearning forget01 ###################################################
########################################################################################################################

# # TOFU unlearning forget01
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/llama2 trainer=GradAscent task_name=llama2_forget01_GradAscent forget_split=forget01 retain_split=retain99
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/llama2 trainer=GradDiff task_name=llama2_forget01_GradDiff forget_split=forget01 retain_split=retain99
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/llama2 trainer=GradDiff trainer.method_args.retain_loss_type=KL task_name=llama2_forget01_GradDiff_KL forget_split=forget01 retain_split=retain99
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/llama2 trainer=NPO task_name=llama2_forget01_NPO forget_split=forget01 retain_split=retain99
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/llama2 trainer=NPO trainer.method_args.retain_loss_type=KL task_name=llama2_forget01_NPO_KL forget_split=forget01 retain_split=retain99
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/llama2_idk trainer=DPO data/datasets@data.forget=TOFU_QA_forget_idk task_name=llama2_forget01_IdkDPO forget_split=forget01 retain_split=retain99