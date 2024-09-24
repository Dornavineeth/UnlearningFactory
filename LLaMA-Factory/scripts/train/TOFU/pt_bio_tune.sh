dataset=TOFU_BIO
num_epochs=10
lrs=(1e-5, 2e-5, 5e-5)
# TODO: Set wamup steps correctly

for lr in "${lrs[@]}"
do
    python src/train.py \
    --stage pt \
    --do_train True \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template llama3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset ${dataset} \
    --cutoff_len 1024 \
    --learning_rate ${lr} \
    --num_train_epochs ${num_epochs} \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --save_strategy epoch \
    --save_total_limit 5 \
    --warmup_steps 5 \
    --weight_decay 0.1 \
    --optim "paged_adamw_32bit" \
    --packing False \
    --report_to none \
    --output_dir ./saves/${dataset}/pt \
    --overwrite_cache \
    --overwrite_output_dir \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True
done