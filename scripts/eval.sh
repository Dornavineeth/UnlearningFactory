# TOFU
python src/eval.py experiment=eval/tofu/llama2.yaml forget_split=forget10 task_name=tofu_forget10_target retain_logs_path=saves/eval/tofu_forget10_retain/TOFU_EVAL.json
python src/eval.py experiment=eval/tofu/llama2.yaml forget_split=forget05 task_name=tofu_forget05_target retain_logs_path=saves/eval/tofu_forget05_retain/TOFU_EVAL.json
python src/eval.py experiment=eval/tofu/llama2.yaml forget_split=forget01 task_name=tofu_forget01_target retain_logs_path=saves/eval/tofu_forget01_retain/TOFU_EVAL.json

# TODO: Set retain model. TOFU doesn't provide retain model for llama2
# python src/eval.py experiment=eval/tofu/llama2.yaml forget_split=forget10 task_name=tofu_forget10_retain
# python src/eval.py experiment=eval/tofu/llama2.yaml forget_split=forget10 task_name=tofu_forget05_retain
# python src/eval.py experiment=eval/tofu/llama2.yaml forget_split=forget10 task_name=tofu_forget01_retain


# MUSE News
python src/eval.py experiment=eval/muse/llama2.yaml data_split=News task_name=muse_news_retain model.model_args.pretrained_model_name_or_path=muse-bench/MUSE-news_retrain
python src/eval.py experiment=eval/muse/llama2.yaml data_split=News task_name=muse_news_target retain_logs_path=saves/eval/muse_news_retain/MUSE_EVAL.json

# MUSE Books
python src/eval.py experiment=eval/muse/llama2.yaml data_split=Books task_name=muse_books_retain model.model_args.pretrained_model_name_or_path=muse-bench/MUSE-books_retrain
python src/eval.py experiment=eval/muse/llama2.yaml data_split=Books task_name=muse_books_target retain_logs_path=saves/eval/muse_books_retain/MUSE_EVAL.json
