python3 08_14_src/encode_for_lora4.py \
    --model_name=llama3.2-1b-instruct \
    --dataset=complexwebquestions \
    --sample=100 \
    --per_device_train_batch_size=2 \
    --num_train_epochs=5 \
    --save_epochs 1 2 3 5 \
    --learning_rate=0.0001 \
    --weight_decay=0.05 \
    --lora_rank=6 \
    --lora_alpha=96

python3 08_14_src/inference_for_lora4.py \
    --model_name=llama3.2-1b-instruct \
    --dataset=complexwebquestions \
    --sample=100 \
    --num_train_epochs=5 \
    --checkpoint_epoch=5 \
    --learning_rate=0.0001 \
    --weight_decay=0.05 \
    --lora_rank=6 \
    --lora_alpha=96 \
    --max_new_tokens=20 \
    --inference_method=lora4_prag