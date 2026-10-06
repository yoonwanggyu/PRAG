import os
import gc
import json
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
    return prompt_ids


def train(question, augments, args, model, tokenizer, 
          init_adapter_path, save_path):
    prompt_ids = get_train_data(args.augment_model, augments, tokenizer, args)
    train_data = TrainingData(prompt_ids, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.per_device_train_batch_size,
        collate_fn=TrainingDataCollator(tokenizer, model.device),
        shuffle=False,
    )
    model = PeftModel.from_pretrained(model, init_adapter_path, is_trainable=True)
    model.is_parallelizable = True
    model.model_parallel = True
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(model_parameters, lr=args.learning_rate)

    loss_log = []
    epoch_avg_losses = []
    for epoch in range(args.num_train_epochs):
        epoch_losses = []
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            epoch_losses.append(loss_value)
            loss_log.append({
                "epoch": epoch,
                "step": step,
                "loss": loss_value,
            })

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_avg_losses.append(avg_epoch_loss)
        print(f"[epoch {epoch}] avg_loss={avg_epoch_loss:.4f}")

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)

    with open(os.path.join(save_path, "train_loss.json"), "w") as f:
        json.dump(loss_log, f, indent=2)

    model = model.unload()
    torch.cuda.empty_cache()
    gc.collect()
    return model, epoch_avg_losses

def append_loss_summary(summary_path, did, pid, epoch_avg_losses):
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "a") as f:
        loss_str = "\t".join(f"{l:.4f}" for l in epoch_avg_losses)
        f.write(f"data_{did}\tpassage_{pid}\t{loss_str}\n")

def compute_epoch_avg_from_json(json_path):
    """저장된 train_loss.json 하나에서 epoch별 평균 loss 리스트를 계산."""
    with open(json_path, "r") as f:
        loss_log = json.load(f)
    epoch_losses = {}
    for entry in loss_log:
        epoch_losses.setdefault(entry["epoch"], []).append(entry["loss"])
    epoch_avg = [
        sum(losses) / len(losses)
        for epoch, losses in sorted(epoch_losses.items())
    ]
    return epoch_avg

def write_overall_summary(output_dir, summary_path):
    """output_dir 아래 모든 data_*/passage_*/train_loss.json을 다시 읽어
    전체 epoch별 평균을 계산하고 summary 파일 맨 아래에 추가."""
    all_epoch_avgs = []
    for did_dir in sorted(os.listdir(output_dir)):
        did_path = os.path.join(output_dir, did_dir)
        if not os.path.isdir(did_path) or not did_dir.startswith("data_"):
            continue
        for pid_dir in sorted(os.listdir(did_path)):
            pid_path = os.path.join(did_path, pid_dir)
            json_path = os.path.join(pid_path, "train_loss.json")
            if os.path.exists(json_path):
                epoch_avg = compute_epoch_avg_from_json(json_path)
                if epoch_avg:
                    all_epoch_avgs.append(epoch_avg)

    if not all_epoch_avgs:
        return

    num_epochs = min(len(x) for x in all_epoch_avgs)
    overall = []
    for e in range(num_epochs):
        vals = [x[e] for x in all_epoch_avgs]
        overall.append(sum(vals) / len(vals))

    with open(summary_path, "a") as f:
        loss_str = "\t".join(f"{l:.4f}" for l in overall)
        f.write(f"OVERALL_AVG\t({len(all_epoch_avgs)}_passages)\t{loss_str}\n")

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
            lora_dropout=0, # !!!
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
        print(f"### Solving {filename} ###")
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

        summary_path = os.path.join(
            ROOT_DIR,
            "offline",
            args.model_name,
            f"rank={args.lora_rank}_alpha={args.lora_alpha}",
            args.dataset,
            f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
            f"aug_model={args.augment_model}",
            "loss_summary",
            f"{filename}_avg_loss.txt",
        )

        fulldata = fulldata if args.sample == -1 else fulldata[:args.sample]
        for did, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            augment = data["augment"]
            for pid in range(len(augment)): # 0,1,2
                save_path = os.path.join(output_dir, f"data_{did}", f"passage_{pid}")
                if os.path.exists(os.path.join(save_path, "adapter_model.safetensors")):
                    continue
                model, epoch_avg_losses = train(data["question"], [augment[pid]], args, model, tokenizer, 
                            init_adapter_path, save_path)
                append_loss_summary(summary_path, did, pid, epoch_avg_losses)

        # 이 filename(dataset)에 속한 모든 passage 학습이 끝난 뒤
        # 저장된 json을 전부 다시 읽어 epoch별 전체 평균을 summary 맨 아래에 추가
        write_overall_summary(output_dir, summary_path)
                

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