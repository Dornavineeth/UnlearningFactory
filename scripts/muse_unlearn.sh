#########################################################
############### MUSE News Unlearning ####################
#########################################################
retain_logs_path=saves/eval/muse_news_retain/MUSE_EVAL.json 

python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/llama2 data_split=News trainer=GradAscent task_name=llama2_news_GradAscent retain_logs_path=${retain_logs_path}
python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/llama2 data_split=News trainer=GradDiff task_name=llama2_news_GradDiff retain_logs_path=${retain_logs_path}
python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/llama2 data_split=News trainer=GradDiff trainer.method_args.retain_loss_type=KL task_name=llama2_news_GradDiff_KL retain_logs_path=${retain_logs_path}
python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/llama2 data_split=News trainer=NPO task_name=llama2_news_NPO retain_logs_path=${retain_logs_path}
python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/llama2 data_split=News trainer=NPO trainer.method_args.retain_loss_type=KL task_name=llama2_news_NPO_KL retain_logs_path=${retain_logs_path}


#########################################################
############### MUSE Books Unlearning ###################
#########################################################
retain_logs_path=saves/eval/muse_books_retain/MUSE_EVAL.json 

python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/llama2 data_split=Books trainer=GradAscent task_name=llama2_books_GradAscent retain_logs_path=${retain_logs_path}
python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/llama2 data_split=Books trainer=GradDiff task_name=llama2_books_GradDiff retain_logs_path=${retain_logs_path}
python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/llama2 data_split=News trainer=GradDiff trainer.method_args.retain_loss_type=KL task_name=llama2_books_GradDiff_KL retain_logs_path=${retain_logs_path}
python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/llama2 data_split=News trainer=NPO task_name=llama2_books_NPO retain_logs_path=${retain_logs_path}
python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/llama2 data_split=News trainer=NPO trainer.method_args.retain_loss_type=KL task_name=llama2_books_NPO_KL retain_logs_path=${retain_logs_path}


# #########################################################
# ########### MUSE News Unlearning Scalability ############
# #########################################################
retain_logs_path=saves/eval/muse_news_retain/MUSE_EVAL.json 

for scal in "forget_1" "forget_2" "forget_3" "forget_4"
do
    # python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/llama2_scal data_split=News trainer=GradAscent task_name=llama2_news_GradAscent_scal_${scal} forget_split=${scal} retain_logs_path=${retain_logs_path}
    # python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/llama2_scal data_split=News trainer=GradDiff task_name=llama2_news_GradDiff_scal_${scal} forget_split=${scal} retain_logs_path=${retain_logs_path}
    # python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/llama2_scal data_split=News trainer=GradDiff trainer.method_args.retain_loss_type=KL task_name=llama2_news_GradDiff_KL_scal_${scal} forget_split=${scal} retain_logs_path=${retain_logs_path}
    python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/llama2_scal data_split=News trainer=NPO task_name=llama2_news_NPO_scal_${scal} forget_split=${scal} retain_logs_path=${retain_logs_path}
    # python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/llama2_scal data_split=News trainer=NPO trainer.method_args.retain_loss_type=KL task_name=llama2_news_NPO_KL_scal_${scal} forget_split=${scal} retain_logs_path=${retain_logs_path}
done

#########################################################
########### MUSE News Unlearning sustainability #########
#########################################################
# NOTE: script written for only one method, please feel free to set the hyper paramters and test sustainability for all methods
retain_logs_path=saves/eval/muse_news_retain/MUSE_EVAL.json 

model_path=muse-bench/MUSE-News_target
for sust in "forget_1" "forget_2" "forget_3" "forget_4"
do
    python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/llama2_sust \
    data_split=News \
    trainer=NPO \
    model.model_args.pretrained_model_name_or_path=${model_path} \
    task_name=llama2_news_NPO_sust_${sust} \
    forget_split=${sust} \
    retain_logs_path=${retain_logs_path}

    model_path=saves/unlearn/llama2_news_NPO_sust_${sust}
done