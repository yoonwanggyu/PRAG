# python3 src/encode.py \
#     --model_name=llama3.2-1b-instruct \
#     --dataset=popqa \
#     --sample=300 \
#     --per_device_train_batch_size=1 \
#     --num_train_epochs=2 \
#     --learning_rate=0.0003 \
#     --lora_rank=2 \
#     --lora_alpha=32 

python3 src/encode.py \
    --model_name=llama3.2-1b-instruct \
    --dataset=popqa \
    --sample=300 \
    --per_device_train_batch_size=5 \
    --num_train_epochs=5 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 

python3 src/inference.py \
    --model_name=llama3.2-1b-instruct \
    --dataset=popqa \
    --sample=300 \
    --num_train_epochs=5 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --max_new_tokens=128 \
    --inference_method=prag \
    --adapter_merge_type=linear