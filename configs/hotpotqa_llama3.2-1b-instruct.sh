python3 08_14_src/encode_for_lora4.py \
    --model_name=llama3.2-1b-instruct \
    --dataset=hotpotqa \
    --sample=100 \
    --per_device_train_batch_size=1 \
    --num_train_epochs=1 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --with_cot

python3 08_14_src/inference_for_lora4.py \
    --model_name=llama3.2-1b-instruct \
    --dataset=hotpotqa \
    --data_type=comparison \
    --sample=100 \
    --num_train_epochs=1 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --max_new_tokens=128 \
    --inference_method=lora4_prag \
    --with_cot