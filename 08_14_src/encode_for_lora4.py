# """
# encode_for_lora4.py

# 기존 encode.py는 문서(passage)마다 개별 LoRA(LoRA1, LoRA2, LoRA3)를 학습한다.
# 이 스크립트는 한 질문에 관련된 문서 전체의 QA를 합쳐 LoRA4(cat merge)를 학습한다.

# 핵심 차이는 딱 하나다.
# get_train_data는 augments를 리스트로 받아 각 문서를 순회하며 프롬프트를 만든다.
# 기존 encode.py는 이 함수에 [augment[pid]]처럼 문서 하나만 감싸서 넘겼기 때문에
# 문서별 개별 학습이 되었다. 여기서는 augment 전체(문서 3개)를 그대로 넘겨서
# 한 번의 학습으로 3개 문서의 QA를 모두 사용하는 LoRA를 만든다.
# get_train_data, train, TrainingData, TrainingDataCollator는 수정 없이 그대로 재사용한다.
# """

# import os
# import gc
# import time
# import argparse
# import torch
# from tqdm import tqdm
# from peft import TaskType, get_peft_model, LoraConfig, PeftModel
# from torch.utils.data import Dataset
# from transformers import DefaultDataCollator
# from typing import Dict, List

# import json

# import prompt_template
# from root_dir_path import ROOT_DIR
# from utils import get_model, load_data

# import numpy as np
# import random

# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)


# class TrainingData(Dataset):
#     ignored_id = -100

#     def __init__(self, prompt_ids, tokenizer, max_length=3000):
#         self.max_length = max_length
#         self.dataset = []
#         pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
#         for input_ids in prompt_ids:
#             labels = input_ids.copy()
#             if len(input_ids) > max_length:
#                 input_ids = input_ids[:max_length]
#                 labels = labels[:max_length]
#             attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
#             input_ids += [pad_token_id] * (max_length - len(input_ids))
#             labels += [self.ignored_id] * (max_length - len(labels))
#             self.dataset.append({
#                 "input_ids": input_ids,
#                 "labels": labels,
#                 "attention_mask": attention_mask,
#             })
#         self.total_len = len(self.dataset)

#     def __len__(self):
#         return self.total_len

#     def __getitem__(self, idx) -> Dict[str, list]:
#         return self.dataset[idx]


# class TrainingDataCollator(DefaultDataCollator):
#     def __init__(self, tokenizer, device):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.device = device

#     def __call__(self, examples: List[Dict[str, list]]) -> Dict[str, torch.Tensor]:
#         input_ids, labels, attention_mask = tuple(
#             map(lambda x: [example[x] for example in examples], ["input_ids", "labels", "attention_mask"])
#         )
#         return {
#             "input_ids": torch.tensor(input_ids).to(self.device),
#             "labels": torch.tensor(labels).to(self.device),
#             "attention_mask": torch.tensor(attention_mask).to(self.device),
#         }


# def get_train_data(aug_model, augments, tokenizer, args):
#     """augments에 문서 여러 개를 담아 넘기면, 각 문서의 QA가 모두 프롬프트로 만들어진다.
#     LoRA4 학습에서는 이 augments에 한 질문의 관련 문서 전체(3개)를 넘긴다.
#     """
#     from prompt_template import get_prompt
#     prompt_ids = []
#     for aug in augments:
#         psg = aug["passage"]
#         rew = aug[f"{aug_model}_rewrite"]
#         qas = aug[f"{aug_model}_qa"]
#         qpa_cnt = (len(qas) + 1) // 2
#         for qid, qa in enumerate(qas):
#             if qid < qpa_cnt:
#                 for ppp in [psg, rew]:
#                     prompt_ids.append(get_prompt(tokenizer, qa["question"],
#                                                     [ppp],
#                                                     qa["answer"] if not args.with_cot else qa["full_answer"],
#                                                     with_cot=args.with_cot))
#             else:
#                 prompt_ids.append(get_prompt(tokenizer, qa["question"],
#                                                 None,
#                                                 qa["answer"] if not args.with_cot else qa["full_answer"],
#                                                 with_cot=args.with_cot))
#     # print("prompt_ids 개수 :" f"{len(prompt_ids)}")
#     return prompt_ids


# # def train(question, augments, args, model, tokenizer,
# #           init_adapter_path, save_path):
# #     prompt_ids = get_train_data(args.augment_model, augments, tokenizer, args)
# #     train_data = TrainingData(prompt_ids, tokenizer)
# #     train_dataloader = torch.utils.data.DataLoader(
# #         train_data,
# #         batch_size=args.per_device_train_batch_size,
# #         collate_fn=TrainingDataCollator(tokenizer, model.device),
# #         shuffle=True,
# #     )
# #     model = PeftModel.from_pretrained(model, init_adapter_path, is_trainable=True)
# #     model.is_parallelizable = True
# #     model.model_parallel = True
# #     model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# #     optimizer = torch.optim.AdamW(model_parameters, lr=args.learning_rate)

# #     loss_log = []
# #     epoch_avg_losses = []
# #     for epoch in range(args.num_train_epochs):
# #         epoch_losses = []
# #         for step, batch in enumerate(train_dataloader):
# #             optimizer.zero_grad()
# #             outputs = model(**batch)
# #             loss = outputs.loss
# #             loss.backward()
# #             optimizer.step()

# #             loss_value = loss.item()
# #             epoch_losses.append(loss_value)
# #             loss_log.append({
# #                 "epoch": epoch,
# #                 "step": step,
# #                 "loss": loss_value,
# #             })

# #         avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
# #         epoch_avg_losses.append(avg_epoch_loss)
# #         print(f"[epoch {epoch}] avg_loss={avg_epoch_loss:.4f}")

# #     os.makedirs(save_path, exist_ok=True)
# #     model.save_pretrained(save_path)

# #     with open(os.path.join(save_path, "train_loss.json"), "w") as f:
# #         json.dump(loss_log, f, indent=2)

# #     model = model.unload()
# #     torch.cuda.empty_cache()
# #     gc.collect()
# #     return model, epoch_avg_losses

# def train(question, augments, args, model, tokenizer,
#           init_adapter_path, save_path):
#     prompt_ids = get_train_data(
#         args.augment_model,
#         augments,
#         tokenizer,
#         args,
#     )
#     train_data = TrainingData(prompt_ids, tokenizer)

#     train_dataloader = torch.utils.data.DataLoader(
#         train_data,
#         batch_size=args.per_device_train_batch_size,
#         collate_fn=TrainingDataCollator(tokenizer, model.device),
#         shuffle=True,
#     )

#     model = PeftModel.from_pretrained(
#         model,
#         init_adapter_path,
#         is_trainable=True,
#     )
#     model.is_parallelizable = True
#     model.model_parallel = True
#     model.train()

#     model_parameters = [
#         p for p in model.parameters()
#         if p.requires_grad
#     ]

#     optimizer = torch.optim.AdamW(
#         model_parameters,
#         lr=args.learning_rate,
#         weight_decay=args.weight_decay,
#     )

#     os.makedirs(save_path, exist_ok=True)

#     save_epochs = set(args.save_epochs)
#     loss_log = []
#     epoch_avg_losses = []

#     for epoch in range(args.num_train_epochs):
#         epoch_number = epoch + 1
#         epoch_losses = []

#         for step, batch in enumerate(train_dataloader):
#             optimizer.zero_grad()

#             outputs = model(**batch)
#             loss = outputs.loss

#             loss.backward()
#             optimizer.step()

#             loss_value = loss.item()
#             epoch_losses.append(loss_value)

#             loss_log.append({
#                 "epoch": epoch,
#                 "epoch_number": epoch_number,
#                 "step": step,
#                 "loss": loss_value,
#             })

#         avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
#         epoch_avg_losses.append(avg_epoch_loss)

#         print(
#             f"[epoch {epoch_number}] "
#             f"avg_loss={avg_epoch_loss:.4f}"
#         )

#         # 지정한 epoch의 LoRA adapter를 각각 저장
#         if epoch_number in save_epochs:
#             checkpoint_path = os.path.join(
#                 save_path,
#                 f"checkpoint_epoch_{epoch_number}",
#             )
#             os.makedirs(checkpoint_path, exist_ok=True)
#             model.save_pretrained(checkpoint_path)

#             print(
#                 f"Saved epoch {epoch_number} checkpoint "
#                 f"to {checkpoint_path}"
#             )

#     # 전체 5 epoch의 loss 기록
#     with open(
#         os.path.join(save_path, "train_loss.json"),
#         "w",
#     ) as f:
#         json.dump(loss_log, f, indent=2)

#     # 정상적으로 학습이 완료됐는지 확인하기 위한 파일
#     with open(
#         os.path.join(save_path, "training_complete.json"),
#         "w",
#     ) as f:
#         json.dump(
#             {
#                 "num_train_epochs": args.num_train_epochs,
#                 "saved_epochs": sorted(save_epochs),
#             },
#             f,
#             indent=2,
#         )

#     model = model.unload()
#     torch.cuda.empty_cache()
#     gc.collect()

#     return model, epoch_avg_losses


# def append_loss_summary(summary_path, did, epoch_avg_losses):
#     os.makedirs(os.path.dirname(summary_path), exist_ok=True)
#     with open(summary_path, "a") as f:
#         loss_str = "\t".join(f"{l:.4f}" for l in epoch_avg_losses)
#         f.write(f"data_{did}\tlora4\t{loss_str}\n")


# def compute_epoch_avg_from_json(json_path):
#     """저장된 train_loss.json 하나에서 epoch별 평균 loss 리스트를 계산."""
#     with open(json_path, "r") as f:
#         loss_log = json.load(f)
#     epoch_losses = {}
#     for entry in loss_log:
#         epoch_losses.setdefault(entry["epoch"], []).append(entry["loss"])
#     epoch_avg = [
#         sum(losses) / len(losses)
#         for epoch, losses in sorted(epoch_losses.items())
#     ]
#     return epoch_avg


# def write_overall_summary(output_dir, summary_path):
#     """output_dir 아래 모든 data_*/lora4/train_loss.json을 다시 읽어
#     전체 epoch별 평균을 계산하고 summary 파일 맨 아래에 추가."""
#     all_epoch_avgs = []
#     for did_dir in sorted(os.listdir(output_dir)):
#         did_path = os.path.join(output_dir, did_dir)
#         if not os.path.isdir(did_path) or not did_dir.startswith("data_"):
#             continue
#         json_path = os.path.join(did_path, "lora4", "train_loss.json")
#         if os.path.exists(json_path):
#             epoch_avg = compute_epoch_avg_from_json(json_path)
#             if epoch_avg:
#                 all_epoch_avgs.append(epoch_avg)

#     if not all_epoch_avgs:
#         return

#     num_epochs = min(len(x) for x in all_epoch_avgs)
#     overall = []
#     for e in range(num_epochs):
#         vals = [x[e] for x in all_epoch_avgs]
#         overall.append(sum(vals) / len(vals))

#     with open(summary_path, "a") as f:
#         loss_str = "\t".join(f"{l:.4f}" for l in overall)
#         f.write(f"OVERALL_AVG\t({len(all_epoch_avgs)}_data)\t{loss_str}\n")


# def main(args):
#     data_list = load_data(args.dataset, args.data_type, args.augment_model)
#     model, tokenizer, _generation_config = get_model(args.model_name)
#     if args.with_cot:
#         prompt_template.get_fewshot(args.dataset)

#     init_adapter_path = os.path.join(
#         ROOT_DIR,
#         "offline",
#         args.model_name,
#         f"rank={args.lora_rank}_alpha={args.lora_alpha}",
#         "base_weight",
#     )
#     if not os.path.exists(os.path.join(init_adapter_path, "adapter_model.safetensors")):
#         print("No LoRA base weight, creating...")
#         peft_config = LoraConfig(
#             task_type=TaskType.CAUSAL_LM,
#             target_modules=['down_proj', 'gate_proj', 'up_proj'],
#             inference_mode=False,
#             r=args.lora_rank,
#             lora_alpha=args.lora_alpha,
#             lora_dropout=0,  # !!!
#         )
#         model = get_peft_model(model, peft_config)
#         model.is_parallelizable = True
#         model.model_parallel = True
#         print(f'Save LoRA base weight to {init_adapter_path}')
#         os.makedirs(init_adapter_path, exist_ok=True)
#         model.save_pretrained(init_adapter_path)
#         model = model.unload()
#         time.sleep(2)
#         assert os.path.exists(os.path.join(init_adapter_path, "adapter_model.safetensors"))

#     cot_name = "cot" if args.with_cot else "direct"
#     for filename, fulldata in data_list:
#         filename = filename.split('.')[0]
#         print(f"### Solving {filename} (LoRA4 cat merge) ###")
#         output_dir = os.path.join(
#             ROOT_DIR,
#             "offline",
#             args.model_name,
#             f"rank={args.lora_rank}_alpha={args.lora_alpha}",
#             args.dataset,
#             f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
#             f"aug_model={args.augment_model}",
#             filename,
#         )
#         os.makedirs(output_dir, exist_ok=True)

#         summary_path = os.path.join(
#             ROOT_DIR,
#             "offline",
#             args.model_name,
#             f"rank={args.lora_rank}_alpha={args.lora_alpha}",
#             args.dataset,
#             f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
#             f"aug_model={args.augment_model}",
#             "loss_summary",
#             f"{filename}_lora4_avg_loss.txt",
#         )

#         fulldata = fulldata if args.sample == -1 else fulldata[:args.sample]
#         for did, data in tqdm(enumerate(fulldata), total=len(fulldata)):
#             augment = data["augment"]
#             # 문서별로 나누지 않고 augment 전체(관련 문서 전부)를 한 번에 학습한다.
#             # 이렇게 만들어진 LoRA가 LoRA4(cat merge)에 해당한다.
#             # save_path = os.path.join(output_dir, f"data_{did}", "lora4")
#             # if os.path.exists(os.path.join(save_path, "adapter_model.safetensors")):
#             #     continue
#             # model, epoch_avg_losses = train(data["question"], augment, args, model, tokenizer,
#             #             init_adapter_path, save_path)
#             # append_loss_summary(summary_path, did, epoch_avg_losses)

#             save_path = os.path.join(
#                         output_dir,
#                         f"data_{did}",
#                         "lora4",
#                     )

#             complete_path = os.path.join(
#                         save_path,
#                         "training_complete.json",
#                     )

#             if os.path.exists(complete_path):
#                         continue

#             model, epoch_avg_losses = train(
#                         data["question"],
#                         augment,
#                         args,
#                         model,
#                         tokenizer,
#                         init_adapter_path,
#                         save_path,
#                     )

#             append_loss_summary(
#                         summary_path,
#                         did,
#                         epoch_avg_losses,
#                     )

#         # 이 filename(dataset)에 속한 모든 data의 lora4 학습이 끝난 뒤
#         # 저장된 json을 전부 다시 읽어 epoch별 전체 평균을 summary 맨 아래에 추가
#         write_overall_summary(output_dir, summary_path)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_name", type=str, required=True)
#     parser.add_argument("--dataset", type=str, required=True)
#     parser.add_argument("--data_type", type=str)
#     parser.add_argument("--with_cot", action="store_true")
#     parser.add_argument("--sample", type=int, default=-1)  # -1 means all
#     parser.add_argument("--augment_model", type=str, default=None)
#     # Train
#     parser.add_argument("--per_device_train_batch_size", type=int, default=1)
#     parser.add_argument("--num_train_epochs", type=int, default=3)
#     parser.add_argument("--save_epochs",type=int,nargs="+",default=[1, 2, 3, 5])
#     parser.add_argument("--learning_rate", type=float, default=3e-4)
#     # LoRA
#     parser.add_argument("--lora_rank", type=int, default=None)
#     parser.add_argument("--lora_alpha", type=int, default=None)
#     args = parser.parse_args()
#     assert args.lora_rank and args.lora_alpha, "No config for LoRA"
#     if args.augment_model is None:
#         args.augment_model = args.model_name
#     print(args)
#     main(args)

"""
encode_for_lora4.py

한 질문과 관련된 여러 문서의 QA를 합쳐 단일 LoRA4를 학습한다.

- 최대 num_train_epochs까지 한 번 학습
- save_epochs에 지정된 epoch checkpoint 저장
- AdamW weight decay를 argument로 제어
"""

import os
import gc
import time
import json
import random
import argparse
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from peft import TaskType, get_peft_model, LoraConfig, PeftModel
from torch.utils.data import Dataset
from transformers import DefaultDataCollator

import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model, load_data


# ------------------------------------------------------------------
# Seed
# ------------------------------------------------------------------

seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class TrainingData(Dataset):
    ignored_id = -100

    def __init__(
        self,
        prompt_ids,
        tokenizer,
        max_length=3000,
    ):
        self.max_length = max_length
        self.dataset = []

        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else 0
        )

        for input_ids in prompt_ids:
            # 원본 prompt_ids를 직접 수정하지 않도록 복사
            input_ids = input_ids.copy()
            labels = input_ids.copy()

            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]

            current_length = len(input_ids)
            padding_length = max_length - current_length

            attention_mask = (
                [1] * current_length
                + [0] * padding_length
            )

            input_ids += [pad_token_id] * padding_length
            labels += [self.ignored_id] * padding_length

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

    def __call__(
        self,
        examples: List[Dict[str, list]],
    ) -> Dict[str, torch.Tensor]:

        input_ids, labels, attention_mask = tuple(
            map(
                lambda key: [
                    example[key]
                    for example in examples
                ],
                [
                    "input_ids",
                    "labels",
                    "attention_mask",
                ],
            )
        )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
                device=self.device,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
                device=self.device,
            ),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
                device=self.device,
            ),
        }


# ------------------------------------------------------------------
# Train data construction
# ------------------------------------------------------------------

def get_train_data(
    aug_model,
    augments,
    tokenizer,
    args,
):
    """
    augments에 포함된 모든 문서의 QA를 하나의 학습 데이터로 만든다.
    """

    from prompt_template import get_prompt

    prompt_ids = []

    for aug in augments:
        passage = aug["passage"]
        rewrite = aug[f"{aug_model}_rewrite"]
        qas = aug[f"{aug_model}_qa"]

        qpa_count = (len(qas) + 1) // 2

        for qid, qa in enumerate(qas):
            answer = (
                qa["full_answer"]
                if args.with_cot
                else qa["answer"]
            )

            if qid < qpa_count:
                for current_passage in [passage, rewrite]:
                    prompt_ids.append(
                        get_prompt(
                            tokenizer,
                            qa["question"],
                            [current_passage],
                            answer,
                            with_cot=args.with_cot,
                        )
                    )
            else:
                prompt_ids.append(
                    get_prompt(
                        tokenizer,
                        qa["question"],
                        None,
                        answer,
                        with_cot=args.with_cot,
                    )
                )

    return prompt_ids


# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------

def train(
    question,
    augments,
    args,
    model,
    tokenizer,
    init_adapter_path,
    save_path,
):
    del question  # 현재 학습 함수에서 직접 사용하지 않음

    prompt_ids = get_train_data(
        args.augment_model,
        augments,
        tokenizer,
        args,
    )

    if not prompt_ids:
        raise ValueError(
            f"No training prompts were generated: {save_path}"
        )

    train_data = TrainingData(
        prompt_ids,
        tokenizer,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.per_device_train_batch_size,
        collate_fn=TrainingDataCollator(
            tokenizer,
            model.device,
        ),
        shuffle=True,
    )

    # 동일한 초기 adapter에서 각 data별 LoRA4 학습 시작
    model = PeftModel.from_pretrained(
        model,
        init_adapter_path,
        is_trainable=True,
    )

    model.is_parallelizable = True
    model.model_parallel = True
    model.train()

    model_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    # AdamW optimizer
    optimizer = torch.optim.AdamW(
        model_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    os.makedirs(save_path, exist_ok=True)

    save_epochs = set(args.save_epochs)
    loss_log = []
    epoch_avg_losses = []

    for epoch_index in range(args.num_train_epochs):
        epoch_number = epoch_index + 1
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
                "epoch": epoch_index,
                "epoch_number": epoch_number,
                "step": step,
                "loss": loss_value,
            })

        if not epoch_losses:
            raise RuntimeError(
                f"No optimizer steps were executed: {save_path}"
            )

        avg_epoch_loss = (
            sum(epoch_losses)
            / len(epoch_losses)
        )

        epoch_avg_losses.append(avg_epoch_loss)

        print(
            f"[epoch {epoch_number}] "
            f"avg_loss={avg_epoch_loss:.4f}"
        )

        # 지정한 epoch의 adapter checkpoint 저장
        if epoch_number in save_epochs:
            checkpoint_path = os.path.join(
                save_path,
                f"checkpoint_epoch_{epoch_number}",
            )

            os.makedirs(
                checkpoint_path,
                exist_ok=True,
            )

            model.save_pretrained(
                checkpoint_path,
            )

            print(
                f"Saved epoch {epoch_number} checkpoint "
                f"to {checkpoint_path}"
            )

    # 전체 step loss 저장
    with open(
        os.path.join(
            save_path,
            "train_loss.json",
        ),
        "w",
    ) as file:
        json.dump(
            loss_log,
            file,
            indent=2,
        )

    # 해당 data의 학습 설정 저장
    with open(
        os.path.join(
            save_path,
            "training_complete.json",
        ),
        "w",
    ) as file:
        json.dump(
            {
                "num_train_epochs": args.num_train_epochs,
                "saved_epochs": sorted(save_epochs),
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "batch_size": args.per_device_train_batch_size,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
            },
            file,
            indent=2,
        )

    # PEFT adapter 제거 후 base model로 복원
    model = model.unload()

    torch.cuda.empty_cache()
    gc.collect()

    return model, epoch_avg_losses


# ------------------------------------------------------------------
# Loss summary
# ------------------------------------------------------------------

def append_loss_summary(
    summary_path,
    did,
    epoch_avg_losses,
):
    os.makedirs(
        os.path.dirname(summary_path),
        exist_ok=True,
    )

    with open(summary_path, "a") as file:
        loss_string = "\t".join(
            f"{loss:.4f}"
            for loss in epoch_avg_losses
        )

        file.write(
            f"data_{did}\tlora4\t{loss_string}\n"
        )


def compute_epoch_avg_from_json(json_path):
    """
    train_loss.json에서 epoch별 평균 loss를 계산한다.
    """

    with open(json_path, "r") as file:
        loss_log = json.load(file)

    epoch_losses = {}

    for entry in loss_log:
        epoch_losses.setdefault(
            entry["epoch"],
            [],
        ).append(entry["loss"])

    return [
        sum(losses) / len(losses)
        for _, losses in sorted(
            epoch_losses.items()
        )
    ]


def write_overall_summary(
    output_dir,
    summary_path,
):
    """
    모든 data_*/lora4/train_loss.json을 읽어
    epoch별 전체 평균 loss를 계산한다.
    """

    all_epoch_avgs = []

    for did_dir in sorted(os.listdir(output_dir)):
        did_path = os.path.join(
            output_dir,
            did_dir,
        )

        if (
            not os.path.isdir(did_path)
            or not did_dir.startswith("data_")
        ):
            continue

        json_path = os.path.join(
            did_path,
            "lora4",
            "train_loss.json",
        )

        if os.path.exists(json_path):
            epoch_avg = compute_epoch_avg_from_json(
                json_path
            )

            if epoch_avg:
                all_epoch_avgs.append(epoch_avg)

    if not all_epoch_avgs:
        return

    num_epochs = min(
        len(epoch_losses)
        for epoch_losses in all_epoch_avgs
    )

    overall = []

    for epoch_index in range(num_epochs):
        values = [
            epoch_losses[epoch_index]
            for epoch_losses in all_epoch_avgs
        ]

        overall.append(
            sum(values) / len(values)
        )

    os.makedirs(
        os.path.dirname(summary_path),
        exist_ok=True,
    )

    with open(summary_path, "a") as file:
        loss_string = "\t".join(
            f"{loss:.4f}"
            for loss in overall
        )

        file.write(
            f"OVERALL_AVG\t"
            f"({len(all_epoch_avgs)}_data)\t"
            f"{loss_string}\n"
        )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(args):
    data_list = load_data(
        args.dataset,
        args.data_type,
        args.augment_model,
    )

    model, tokenizer, _generation_config = get_model(
        args.model_name
    )

    if args.with_cot:
        prompt_template.get_fewshot(
            args.dataset
        )

    # Rank와 alpha에 따라 초기 LoRA adapter가 달라진다.
    # Weight decay는 optimizer 설정이므로 base_weight와 무관하다.
    init_adapter_path = os.path.join(
        ROOT_DIR,
        "offline",
        args.model_name,
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        "base_weight",
    )

    init_adapter_file = os.path.join(
        init_adapter_path,
        "adapter_model.safetensors",
    )

    if not os.path.exists(init_adapter_file):
        print("No LoRA base weight, creating...")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "down_proj",
                "gate_proj",
                "up_proj",
            ],
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
        )

        model = get_peft_model(
            model,
            peft_config,
        )

        model.is_parallelizable = True
        model.model_parallel = True

        print(
            f"Save LoRA base weight to "
            f"{init_adapter_path}"
        )

        os.makedirs(
            init_adapter_path,
            exist_ok=True,
        )

        model.save_pretrained(
            init_adapter_path
        )

        # PEFT 중복 wrapping 방지
        model = model.unload()

        time.sleep(2)

        assert os.path.exists(init_adapter_file), (
            f"Failed to create base adapter: "
            f"{init_adapter_file}"
        )

    cot_name = (
        "cot"
        if args.with_cot
        else "direct"
    )

    # Weight decay를 경로에 포함해 실험 결과 분리
    train_config_dir = (
        f"lr={args.learning_rate}"
        f"_epoch={args.num_train_epochs}"
        f"_wd={args.weight_decay}"
        f"_{cot_name}"
    )

    for filename, fulldata in data_list:
        filename = filename.split(".")[0]

        print(
            f"### Solving {filename} "
            f"(LoRA4 cat merge) ###"
        )

        output_dir = os.path.join(
            ROOT_DIR,
            "offline",
            args.model_name,
            f"rank={args.lora_rank}_alpha={args.lora_alpha}",
            args.dataset,
            train_config_dir,
            f"aug_model={args.augment_model}",
            filename,
        )

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        summary_path = os.path.join(
            ROOT_DIR,
            "offline",
            args.model_name,
            f"rank={args.lora_rank}_alpha={args.lora_alpha}",
            args.dataset,
            train_config_dir,
            f"aug_model={args.augment_model}",
            "loss_summary",
            f"{filename}_lora4_avg_loss.txt",
        )

        if args.sample != -1:
            fulldata = fulldata[:args.sample]

        for did, data in tqdm(
            enumerate(fulldata),
            total=len(fulldata),
        ):
            augment = data["augment"]

            save_path = os.path.join(
                output_dir,
                f"data_{did}",
                "lora4",
            )

            complete_path = os.path.join(
                save_path,
                "training_complete.json",
            )

            if os.path.exists(complete_path):
                continue

            model, epoch_avg_losses = train(
                data["question"],
                augment,
                args,
                model,
                tokenizer,
                init_adapter_path,
                save_path,
            )

            append_loss_summary(
                summary_path,
                did,
                epoch_avg_losses,
            )

        write_overall_summary(
            output_dir,
            summary_path,
        )


# ------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--data_type",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--with_cot",
        action="store_true",
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--augment_model",
        type=str,
        default=None,
    )

    # Train
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--save_epochs",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5],
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )

    # LoRA
    parser.add_argument(
        "--lora_rank",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--lora_alpha",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    if args.augment_model is None:
        args.augment_model = args.model_name

    if args.num_train_epochs <= 0:
        raise ValueError(
            "--num_train_epochs must be greater than 0"
        )

    invalid_save_epochs = [
        epoch
        for epoch in args.save_epochs
        if epoch < 1
        or epoch > args.num_train_epochs
    ]

    if invalid_save_epochs:
        raise ValueError(
            "All --save_epochs values must be between "
            f"1 and {args.num_train_epochs}. "
            f"Invalid values: {invalid_save_epochs}"
        )

    if args.weight_decay < 0:
        raise ValueError(
            "--weight_decay must be greater than or equal to 0"
        )

    print(args)

    main(args)