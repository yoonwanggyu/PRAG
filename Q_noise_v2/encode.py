import os,json,math
import gc
import time
import argparse
import torch
from tqdm import tqdm
from peft import TaskType, get_peft_model, LoraConfig, PeftModel
from torch.utils.data import Dataset
from transformers import DefaultDataCollator
from typing import Dict, List

import prompt_template
from utils import get_model

import numpy as np
import random

import os, json, math, gc, time
import torch
from peft import TaskType, get_peft_model, LoraConfig, PeftModel
from torch.utils.data import Dataset
from transformers import DefaultDataCollator


seed = 111
torch.manual_seed(seed)

# ==========================================
# 1. 데이터셋 & 콜레이터 (원형 유지)
# ==========================================
class TrainingData(Dataset):
    ignored_id = -100
    def __init__(self, prompt_ids, tokenizer, max_length=3000):
        self.max_length = max_length
        self.dataset = []
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        for input_ids in prompt_ids:
            labels = input_ids.copy()
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
            attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
            input_ids += [pad_token_id] * (max_length - len(input_ids))
            labels += [self.ignored_id] * (max_length - len(labels))
            self.dataset.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            })
        self.total_len = len(self.dataset)
    
    def __len__(self): return self.total_len
    def __getitem__(self, idx): return self.dataset[idx]

class TrainingDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
    
    def __call__(self, examples):
        input_ids, labels, attention_mask = tuple(
            map(lambda x: [example[x] for example in examples], ["input_ids", "labels", "attention_mask"])
        )
        return {
            "input_ids": torch.tensor(input_ids).to(self.device),
            "labels": torch.tensor(labels).to(self.device),
            "attention_mask": torch.tensor(attention_mask).to(self.device),
        }

# ==========================================
# 2. 프롬프트 생성 (키 이름만 이전 JSON에 맞춤)
# ==========================================
def get_train_data(aug_data, model_name, tokenizer, with_cot=False):
    from prompt_template import get_prompt
    prompt_ids = []
    
    # 이전 단계에서 만든 JSON의 키 값과 동일하게 매칭
    psg = aug_data["original_passage"] 
    rew = aug_data[f"{model_name}_rewrite"]
    qas = aug_data[f"{model_name}_qa"]
    
    qpa_cnt = (len(qas) + 1) // 2
    for qid, qa in enumerate(qas):
        if qid < qpa_cnt:
            for ppp in [psg, rew]:
                prompt_ids.append(get_prompt(tokenizer, qa["question"], [ppp], qa["answer"], with_cot=with_cot))
        else:
            prompt_ids.append(get_prompt(tokenizer, qa["question"], None, qa["answer"], with_cot=with_cot))
            
    return prompt_ids

# ==========================================
# 3. 핵심 학습 함수
# ==========================================
def train_lora_for_passage(aug_data, model_name, model, tokenizer, init_adapter_path, save_path, lr, epochs, batch_size, with_cot):
    prompt_ids = get_train_data(aug_data, model_name, tokenizer, with_cot)
    train_data = TrainingData(prompt_ids, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        collate_fn=TrainingDataCollator(tokenizer, model.device),
        shuffle=False,
    )

    # 깡통 Base LoRA 장착 및 학습 가능 상태로 변경
    model = PeftModel.from_pretrained(model, init_adapter_path, is_trainable=True)
    model.is_parallelizable = True
    model.model_parallel = True
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(model_parameters, lr=lr)

    # ======================================
    # Loss 기록용 변수 세팅
    # ======================================
    epoch_loss_log = {}
    num_samples = len(prompt_ids)
    steps_per_epoch = math.ceil(num_samples / batch_size)

    # 학습 루프
    model.train()
    for epoch in range(epochs):
        step_losses = []
        
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            # 현재 스텝의 loss 값을 리스트에 추가
            step_losses.append(loss.detach().cpu().item())
            
        # 에포크가 끝날 때마다 딕셔너리에 저장
        epoch_loss_log[f"epoch_{epoch+1}"] = step_losses
        
        # 터미널에서 눈으로 확인할 수 있게 출력
        avg_loss = sum(step_losses) / len(step_losses) if step_losses else 0
        print(f"      Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

    # ======================================
    # 문서 전용 LoRA 저장 및 loss.json 저장
    # ======================================
    os.makedirs(save_path, exist_ok=True)
    
    loss_data = {
        "meta": {
            "num_epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": steps_per_epoch * epochs,
            "learning_rate": lr
        },
        "loss": epoch_loss_log
    }

    with open(os.path.join(save_path, "loss.json"), "w", encoding="utf-8") as f:
        json.dump(loss_data, f, indent=2)

    model.save_pretrained(save_path)
    
    # 다음 학습을 위해 Base 상태로 복귀 및 메모리 정리
    model = model.unload()
    torch.cuda.empty_cache()
    gc.collect()
    
    return model

# ==========================================
# 4. 메인 실행부
# ==========================================
if __name__ == "__main__":
    MODEL_NAME = "llama3.2-1b-instruct" 
    JSON_DATA_PATH = "data_aug_result.json" 
    OUTPUT_BASE_DIR = "./trained_LORA" # LoRA가 저장될 폴더
    
    LORA_RANK = 2
    LORA_ALPHA = 32
    
    # 1. Base 모델 로드
    print("### Loading Base Model ###")
    model, tokenizer, _ = get_model(MODEL_NAME)
    
    # 2. Base LoRA(초기 가중치) 생성 및 저장
    init_adapter_path = os.path.join(OUTPUT_BASE_DIR, "base_weight")
    
    if not os.path.exists(os.path.join(init_adapter_path, "adapter_model.safetensors")):
        print("### Creating Base LoRA Weight ###")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=['down_proj', 'gate_proj', 'up_proj'],
            inference_mode=False,
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0,
        )
        model = get_peft_model(model, peft_config)
        os.makedirs(init_adapter_path, exist_ok=True)
        model.save_pretrained(init_adapter_path)
        
        # 다시 Base 모델로 복귀
        model = model.unload()
        time.sleep(2)
        print("### Base LoRA saved successfully ###")

    # 3. JSON 데이터 로드
    with open(JSON_DATA_PATH, "r", encoding="utf-8") as f:
        augmented_data_list = json.load(f)

    print(f"### Start Training for {len(augmented_data_list)} passages ###")

    import prompt_template
    prompt_template.get_fewshot("2wikimultihopqa")
    
    # 4. 문서별 LoRA 학습 반복
    for data in augmented_data_list:
        pid = data["pid"]
        save_path = os.path.join(OUTPUT_BASE_DIR, f"passage_{pid}_lora")

        is_cot = True if pid == 0 else False
        
        # 이미 학습된 LoRA면 패스
        if os.path.exists(os.path.join(save_path, "adapter_model.safetensors")):
            print(f"Passage {pid} LoRA already exists. Skipping...")
            continue
            
        print(f"--- Training LoRA for Passage {pid} ---")
        model = train_lora_for_passage(
            aug_data=data,
            model_name=MODEL_NAME, 
            model=model,
            tokenizer=tokenizer,
            init_adapter_path=init_adapter_path,
            save_path=save_path,
            lr=0.0003,
            epochs=1,
            batch_size=1,
            with_cot=is_cot
        )
        print(f"Passage {pid} LoRA saved at {save_path}")
        
    print("### All Training Completed! ###")