"""
inference_for_lora4.py

기존 inference.py는 문서별 LoRA(passage_0, passage_1, passage_2)를 각각 불러와
add_weighted_adapter(combination_type="cat")로 파라미터 레벨에서 병합한 뒤 추론한다.

LoRA4는 학습 시점에 이미 데이터 레벨(QA 결합)에서 합쳐진 단일 어댑터이므로,
추론 시점에 별도 병합 과정이 필요 없다. encode_for_lora4.py가 저장한
data_{test_id}/lora4 어댑터 하나만 불러와 바로 추론한다.

inference_method는 icl, lora4_prag, lora4_combine 세 가지를 쓴다.
output_root_dir이 inference_method를 폴더명으로 쓰기 때문에,
기존 prag, combine 결과와 폴더가 겹치지 않고 나란히 비교할 수 있다.
"""

import os
import gc
import json
import argparse
import torch
from tqdm import tqdm
from peft import PeftModel

import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model, evaluate, predict, load_data, read_complete


def main(args):
    data_list = load_data(args.dataset, args.data_type, args.augment_model)
    model, tokenizer, generation_config = get_model(
        args.model_name,
        max_new_tokens=args.max_new_tokens,
    )
    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)

    cot_name = "cot" if args.with_cot else "direct"
    load_adapter_path = os.path.join(
        ROOT_DIR,
        "offline",
        args.model_name,
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
    )
    output_root_dir = os.path.join(
        ROOT_DIR,
        "output",
        args.model_name,
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
        args.inference_method,
    )
    for filename, fulldata in data_list:
        filename = filename.split(".")[0]
        print(f"### Solving {filename} (LoRA4) ###")
        output_dir = os.path.join(output_root_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w") as fout:
            json.dump(vars(args), fout, indent=4)

        predict_file = os.path.join(output_dir, "predict.json")
        ret, start_with = read_complete(predict_file)

        fulldata = fulldata[start_with:] if args.sample == -1 else fulldata[start_with:args.sample]
        for test_id, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            test_id = test_id + start_with
            assert test_id == len(ret), f"test_id {test_id} != len(ret) {len(ret)}"

            question = data["question"]
            passages = data["passages"]
            answer = data["answer"]

            def get_pred(model, psgs):
                text = predict(model, tokenizer, generation_config,
                                        question, with_cot=args.with_cot,
                                        passages=psgs)
                pred = {
                    "test_id": test_id,
                    "question": question,
                    "answer": answer,
                    "text": text,
                }
                pred.update(evaluate(text, answer, args.with_cot))
                return pred

            if args.inference_method == "icl":
                ret.append(get_pred(model, psgs=passages))
            else:
                # 문서별 어댑터를 병합하지 않고, 이미 합쳐서 학습된 lora4 어댑터 하나만 로드한다.
                adapter_path = os.path.join(load_adapter_path, filename, f"data_{test_id}", "lora4")
                model = PeftModel.from_pretrained(
                    model,
                    adapter_path,
                    adapter_name="lora4",
                    is_trainable=False,
                )
                model.set_adapter("lora4")
                ret.append(get_pred(model, psgs=None if args.inference_method == "lora4_prag" else passages))
                model.delete_adapter("lora4")
                model = model.unload()
                torch.cuda.empty_cache()
                gc.collect()

        with open(predict_file, "w") as fout:
            json.dump(ret, fout, indent=4)

        ##### Evaluating #####
        metrics = ["em", "f1", "prec", "recall"]
        ret_str = ""
        for met in metrics:
            acc = sum(float(d[met]) for d in ret) / len(ret)
            acc = round(acc, 4)
            ret_str += f"{met}\t{acc}\n"
        ret_str += "\n" + json.dumps(vars(args), indent=4)
        with open(os.path.join(output_dir, "result.txt"), "w") as fout:
            fout.write(ret_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--sample", type=int, default=-1)  # -1 means all
    parser.add_argument("--augment_model", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--inference_method", type=str, required=True,
                         choices=["icl", "lora4_prag", "lora4_combine"])
    # LoRA
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    args = parser.parse_args()
    assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)