python3 re_src/encode.py \
    --model_name=llama3.2-1b-instruct \
    --dataset=popqa \
    --data_type=total \
    --sample=1 \
    --per_device_train_batch_size=1 \
    --num_train_epochs=2 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 

python3 src/inference.py \
    --model_name=llama3.2-1b-instruct \
    --dataset=popqa \
    --sample=300 \
    --num_train_epochs=2 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --max_new_tokens=20 \
    --inference_method=combine 