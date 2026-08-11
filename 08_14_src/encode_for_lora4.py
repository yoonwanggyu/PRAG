"""
encode_for_lora4.py

기존 encode.py는 문서(passage)마다 개별 LoRA(LoRA1, LoRA2, LoRA3)를 학습한다.
이 스크립트는 한 질문에 관련된 문서 전체의 QA를 합쳐 LoRA4(cat merge)를 학습한다.

핵심 차이는 딱 하나다.
get_train_data는 augments를 리스트로 받아 각 문서를 순회하며 프롬프트를 만든다.
기존 encode.py는 이 함수에 [augment[pid]]처럼 문서 하나만 감싸서 넘겼기 때문에
문서별 개별 학습이 되었다. 여기서는 augment 전체(문서 3개)를 그대로 넘겨서
한 번의 학습으로 3개 문서의 QA를 모두 사용하는 LoRA를 만든다.
get_train_data, train, TrainingData, TrainingDataCollator는 수정 없이 그대로 재사용한다.
"""

import os
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
    """augments에 문서 여러 개를 담아 넘기면, 각 문서의 QA가 모두 프롬프트로 만들어진다.
    LoRA4 학습에서는 이 augments에 한 질문의 관련 문서 전체(3개)를 넘긴다.
    """
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
    # print("prompt_ids 개수 :" f"{len(prompt_ids)}")
    return prompt_ids


def train(question, augments, args, model, tokenizer,
          init_adapter_path, save_path):
    prompt_ids = get_train_data(args.augment_model, augments, tokenizer, args)
    train_data = TrainingData(prompt_ids, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.per_device_train_batch_size,
        collate_fn=TrainingDataCollator(tokenizer, model.device),
        shuffle=True,
    )
    model = PeftModel.from_pretrained(model, init_adapter_path, is_trainable=True)
    model.is_parallelizable = True
    model.model_parallel = True
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(model_parameters, lr=args.learning_rate)
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    model = model.unload()
    torch.cuda.empty_cache()
    gc.collect()
    return model


def main(args):
    data_list = load_data(args.dataset, args.data_type, args.augment_model)
    model, tokenizer, _generation_config = get_model(args.model_name)
    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)

    init_adapter_path = os.path.join(
        ROOT_DIR,
        "offline",
        args.model_name,
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        "base_weight",
    )
    if not os.path.exists(os.path.join(init_adapter_path, "adapter_model.safetensors")):
        print("No LoRA base weight, creating...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=['down_proj', 'gate_proj', 'up_proj'],
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0,  # !!!
        )
        model = get_peft_model(model, peft_config)
        model.is_parallelizable = True
        model.model_parallel = True
        print(f'Save LoRA base weight to {init_adapter_path}')
        os.makedirs(init_adapter_path, exist_ok=True)
        model.save_pretrained(init_adapter_path)
        time.sleep(2)
        assert os.path.exists(os.path.join(init_adapter_path, "adapter_model.safetensors"))

    cot_name = "cot" if args.with_cot else "direct"
    for filename, fulldata in data_list:
        filename = filename.split('.')[0]
        print(f"### Solving {filename} (LoRA4 cat merge) ###")
        output_dir = os.path.join(
            ROOT_DIR,
            "offline",
            args.model_name,
            f"rank={args.lora_rank}_alpha={args.lora_alpha}",
            args.dataset,
            f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
            f"aug_model={args.augment_model}",
            filename,
        )
        os.makedirs(output_dir, exist_ok=True)
        fulldata = fulldata if args.sample == -1 else fulldata[:args.sample]
        for did, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            augment = data["augment"]
            # 문서별로 나누지 않고 augment 전체(관련 문서 전부)를 한 번에 학습한다.
            # 이렇게 만들어진 LoRA가 LoRA4(cat merge)에 해당한다.
            save_path = os.path.join(output_dir, f"data_{did}", "lora4")
            if os.path.exists(os.path.join(save_path, "adapter_model.safetensors")):
                continue
            model = train(data["question"], augment, args, model, tokenizer,
                        init_adapter_path, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--sample", type=int, default=-1)  # -1 means all
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