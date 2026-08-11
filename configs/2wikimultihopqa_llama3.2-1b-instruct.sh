python3 08_14_src/encode_for_lora4.py \
    --model_name=llama3.2-1b-instruct \
    --dataset=2wikimultihopqa \
    --data_type=compositional \
    --sample=300 \
    --per_device_train_batch_size=1 \
    --num_train_epochs=1 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --with_cot

python3 08_14_src/inference_for_lora4.py \
    --model_name=llama3.2-1b-instruct \
    --dataset=2wikimultihopqa \
    --data_type=compositional \
    --sample=300 \
    --num_train_epochs=1 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --max_new_tokens=128 \
    --inference_method=lora4_prag \
    --with_cot