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
from root_dir_path import ROOT_DIR
from utils import get_model, load_data

import numpy as np
import random

seed = 42 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


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
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx) -> Dict[str, list]:
        return self.dataset[idx]


class TrainingDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
    
    def __call__(self, examples: List[Dict[str, list]]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask = tuple(
            map(lambda x: [example[x] for example in examples], ["input_ids", "labels", "attention_mask"])
        )
        return {
            "input_ids": torch.tensor(input_ids).to(self.device),
            "labels": torch.tensor(labels).to(self.device),
            "attention_mask": torch.tensor(attention_mask).to(self.device),
        }
    

def get_train_data(aug_model, augments, tokenizer, args):
    from prompt_template import get_prompt
    prompt_ids = []
    for aug in augments:
        psg = aug["passage"]
        rew = aug[f"{aug_model}_rewrite"]
        qas = aug[f"{aug_model}_qa"]
        qpa_cnt = (len(qas) + 1) // 2
        for qid, qa in enumerate(qas):
            if qid < qpa_cnt:
                for ppp in [psg, rew]:
                    prompt_ids.append(get_prompt(tokenizer, qa["question"], 
                                                    [ppp], 
                                                    qa["answer"] if not args.with_cot else qa["full_answer"], 
                                                    with_cot=args.with_cot))
            else:
                prompt_ids.append(get_prompt(tokenizer, qa["question"], 
                                                None, 
                                                qa["answer"] if not args.with_cot else qa["full_answer"], 
                                                with_cot=args.with_cot))
   
    # 1개의 passage에 대한 LoRA를 학습할 때의 sample 수는 5개
    return prompt_ids


def train(question, augments, args, model, tokenizer, 
          init_adapter_path, save_path):
    prompt_ids = get_train_data(args.augment_model, augments, tokenizer, args)
    train_data = TrainingData(prompt_ids, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.per_device_train_batch_size,
        # batch_size 설정 
        # 1개 LoRA를 학습할 때 한 step에 몇 샘플을 넣느냐
        # 1개의 passage에서 만들어지는 prompt 개수가 상한. 지금 구조에선 batch=5가 이미 최적
        collate_fn=TrainingDataCollator(tokenizer, model.device),
        shuffle=False,
    )
    # ======================================
    # 7) base LoRA weight를 다시 불러와서
    #    "학습 가능한 상태"로 붙임
    # ======================================
    # 이 시점 model은:
    #   [base LLaMA + base LoRA]
        # [핵심] 여기서 합체가 일어납니다!
        # main에서 넘겨받은 건 '순정 model'이지만,
        # 여기서 아까 저장해둔 '깡통 LoRA 파일(init_adapter_path)'을 불러와 입힙니다.
    model = PeftModel.from_pretrained(model, init_adapter_path, is_trainable=True)
    model.is_parallelizable = True
    model.model_parallel = True
    # LoRA 파라미터만 학습 대상
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(model_parameters, lr=args.learning_rate)
    # ======================================
    # 8) 실제 학습
    # ======================================
    epoch_loss_log = {}  # 아래 구조처럼 저장
    num_samples = len(prompt_ids)  
    steps_per_epoch = math.ceil(num_samples / args.per_device_train_batch_size)
    # {
    #     "meta": {
    #         "num_epochs": 5,
    #         "steps_per_epoch": 1,
    #         "total_steps": 5
    #     },
    #     "loss": {
    #         "epoch_1": [1.68, 1.52, 1.31, 1.12, 0.98],
    #         "epoch_2": [0.91, 0.83, 0.74, 0.66, 0.59],
    #         "epoch_3": [0.54, 0.49, 0.44, 0.40, 0.37]
    #     }
    # }

    for epoch in range(args.num_train_epochs):
        step_losses = []

        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # loss 저장 추가
            step_losses.append(loss.detach().cpu().item())
        epoch_loss_log[f"epoch_{epoch+1}"] = step_losses

    # ======================================
    # passage 폴더 생성
    # ======================================
    os.makedirs(save_path, exist_ok=True)

    # ======================================
    # loss.json 저장 
    # ======================================
    loss_data = {
        "meta": {
            "num_epochs": args.num_train_epochs,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": steps_per_epoch * args.num_train_epochs,
            "learning_rate": args.learning_rate
        },
        "loss": epoch_loss_log
    }

    with open(os.path.join(save_path, "loss.json"), "w") as f:
        json.dump(loss_data, f, indent=2)

    # ======================================
    # 9) passage 전용 LoRA 저장
    # ======================================
    # - base model은 저장 안 됨
    # - 오직 LoRA adapter만 저장됨
    model.save_pretrained(save_path)
    # ======================================
    # 10) LoRA 분리 (base model만 남김)
    # ======================================
    # 다음 passage 학습을 위해 깨끗한 상태로 되돌림
    model = model.unload()
    torch.cuda.empty_cache()
    gc.collect()
    return model

# Base LLaMA를 한 번만 메모리에 올려두고, LoRA adapter를 passage마다 붙였다 떼면서 각 passage 전용 LoRA를 따로 학습/저장
# - Base model은 절대 저장 안 함
# - LoRA만 계속 새로 붙이고 저장
# - init_adapter_path = 모든 passage의 공통 출발점
# - save_path = passage-specific skill

def main(args):
    # ======================================
    # 1) Base LLM 로드 (LoRA 없음)
    # ======================================
    # get_model() 내부에서:
    # - AutoModelForCausalLM.from_pretrained(...)
    # - 여기서는 순수 base model (LLaMA-3.2-1B)
    # - 아직 LoRA adapter는 전혀 붙어있지 않음

    data_list = load_data(args.dataset, args.data_type, args.augment_model)
    print(f"data_list 길이 : {len(data_list)}")
    model, tokenizer, _generation_config = get_model(args.model_name)
    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)

    # ======================================
    # 2) "base LoRA weight" 저장 경로
    # ======================================
    # 이 경로에는:
    # - 랜덤 초기화된 LoRA weight가 저장됨
    # - 이후 모든 passage 학습의 출발점이 됨
    target_layers_str = "layers_10-15"

    init_adapter_path = os.path.join(
        ROOT_DIR, 
        "offline", 
        args.model_name, 
        f"rank={args.lora_rank}_alpha={args.lora_alpha}_{target_layers_str}",
        "base_weight",
    )
    # ======================================
    # 3) base LoRA가 없으면 새로 생성
    # ======================================

    if not os.path.exists(os.path.join(init_adapter_path, "adapter_model.safetensors")):
        print("No LoRA base weight, creating...")
        # LoRA 설정 정의
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            # LLaMA의 FFN(MLP) 부분에만 LoRA 적용
            target_modules=['down_proj', 'gate_proj', 'up_proj'],
            # 후반 layer에서만 학습 진행
            layers_to_transform=[10,11,12, 13, 14, 15],
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0, # !!!
        )
        # ==================================
        # (중요) base model에 LoRA 구조를 "붙임"
        # ==================================
        # - 이 시점에서 model은:
        #   [base LLaMA + LoRA adapter]
        # - LoRA weight는 아직 학습되지 않은 초기 상태
        model = get_peft_model(model, peft_config)
        # 병렬화 플래그 (실제로는 single GPU)
        model.is_parallelizable = True
        model.model_parallel = True
        print(f'Save LoRA base weight to {init_adapter_path}')
        # ==================================
        # 4) "초기 LoRA weight" 저장
        # ==================================
        # - 이 weight는 모든 passage 학습의 시작점
        # - 이후 PeftModel.from_pretrained()로 계속 불러와서 사용
        os.makedirs(init_adapter_path, exist_ok=True)
        model.save_pretrained(init_adapter_path)

        # [추가된 코드] 다시 돌아가! (Peft -> Base)
        model = model.unload()
        time.sleep(2)
        assert os.path.exists(os.path.join(init_adapter_path, "adapter_model.safetensors")) 

    cot_name = "cot" if args.with_cot else "direct"
    # ======================================
    # 5) 데이터 단위 반복
    # ======================================
    for filename, fulldata in data_list:
        filename = filename.split('.')[0] 
        print(f"### Solving {filename} ###")
        output_dir = os.path.join(
            ROOT_DIR, 
            "offline", 
            args.model_name, 
            f"rank={args.lora_rank}_alpha={args.lora_alpha}_{target_layers_str}",
            args.dataset,
            f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
            f"aug_model={args.augment_model}",
            filename,
        )
        os.makedirs(output_dir, exist_ok=True)
        fulldata = fulldata if args.sample == -1 else fulldata[:args.sample]
        for did, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            augment = data["augment"]
            # passage 하나씩 처리
            for pid in range(len(augment)):
                save_path = os.path.join(output_dir, f"data_{did}", f"passage_{pid}")
                # 이미 학습된 passage면 skip
                if os.path.exists(os.path.join(save_path, "adapter_model.safetensors")):
                    continue
                # ==================================
                # 6) passage 단위 학습 시작
                # ==================================
                model = train(data["question"], 
                            [augment[pid]], 
                            args, 
                            model, 
                            tokenizer, 
                            init_adapter_path, # base LoRA weight
                            save_path)         # passage-specific LoRA 저장 위치
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--sample", type=int, default=-1) # -1 means all
    parser.add_argument("--augment_model", type=str, default=None)
    # Train
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    args = parser.parse_args()
    assert args.lora_rank and args.lora_alpha, "No config for LoRA"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)