'''
개별 LoRA 부착해서 Upper bound 성능 측정하는 코드
'''


import os
import re
import json
import torch
import string
import gc
import glob
from tqdm import tqdm
from collections import Counter
from typing import List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import evaluate
from prompt_template import get_prompt

def predict(model, tokenizer, generation_config, question, with_cot=None, passages = None):
    model.eval()
    input_ids = get_prompt(
        tokenizer, 
        question, 
        passages = passages, 
        with_cot = with_cot)
    input_len = len(input_ids)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            attention_mask = torch.ones(input_ids.shape).to(model.device),
            **generation_config)
    output = output.sequences[0][input_len:]
    text = tokenizer.decode(output, skip_special_tokens=True)
    return text

def load_json_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception as e:
        print(f"에러가 발생했습니다 ({filepath}): {e}")
        return None

def main():
    import numpy as np 

    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    DATASETS = ["2wikimultihopqa", "complexwebquestions", "hotpotqa", "popqa"]
    WITH_COT = False

    QA_DIR = os.path.join("re_src", "test_qa") 
    OUTPUT_ROOT_DIR = os.path.join("re_src", "output_eval_rank8")
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

    QA_KEY = "generated_questions" 

    print(f"Base 모델 로드 중: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto", 
        trust_remote_code=True
    )
    base_model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    generation_config = dict(
        num_beams=1, 
        do_sample=False,
        max_new_tokens=50,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
    )

    for dataset in DATASETS:
        print(f"\n{'='*50}\n Dataset: {dataset} 평가 시작\n{'='*50}")

        if dataset in ["2wikimultihopqa", "hotpotqa"]:
            cot_name = "cot"
        else:
            cot_name = "direct"

        search_pattern = os.path.join(QA_DIR, f"{dataset}_*questions.json")
        json_files = glob.glob(search_pattern)

        if not json_files:
            print(f"파일 없음 스킵: {search_pattern}")
            break

        for file_path in json_files:
            file_name = os.path.basename(file_path) 
            prefix = file_name.replace(f"{dataset}_", "").replace("_questions.json", "")
            
            print(f"\n처리 시작: [{dataset}] 데이터셋의 [{prefix}] (파일명: {file_name})")

            qa_data_list = load_json_file(file_path)
            if not qa_data_list: break

            dataset_out_dir = os.path.join(OUTPUT_ROOT_DIR, dataset, prefix)
            os.makedirs(dataset_out_dir, exist_ok=True)

            final_json_results = []
            base_all_metrics = {"em": [], "f1": [], "prec": [], "recall": []}
            lora_all_metrics = {"em": [], "f1": [], "prec": [], "recall": []}

            for did, item in enumerate(tqdm(qa_data_list, desc=f"Evaluating {prefix}")):

                if did != 0:
                    continue 

                qa_pairs = item.get(QA_KEY, [])
                if not qa_pairs: 
                    print(f"경고: data_{did}에 '{QA_KEY}' 키가 없거나 비어있습니다.")
                    continue

                # -------------------------------------------------
                # [Step 1] Base 모델 평가
                # -------------------------------------------------
                base_results = []
                for qa in qa_pairs:
                    question = qa["question"]
                    ground_truth = qa["answer"] 
                    
                    pred_text = predict(base_model, tokenizer, generation_config, question, with_cot=WITH_COT, passages=None)
                    metrics = evaluate(pred_text, ground_truth)
                    
                    for k in base_all_metrics: base_all_metrics[k].append(metrics[k])
                    base_results.append({"question": question, "ground_truth": ground_truth, "prediction": pred_text, **metrics})

                # -------------------------------------------------
                # [Step 2] LoRA 어댑터 로드 및 평가
                # -------------------------------------------------
                lora_path = os.path.join(
                    "offline", "llama3.2-1b-instruct", "rank=8_alpha=32", 
                    dataset, f"lr=0.0003_epoch=3_{cot_name}", "aug_model=llama3.2-1b-instruct", 
                    prefix, "merged_passage", "data_0"
                )

                if not os.path.exists(lora_path):
                    print(f"\nLoRA 경로 없음 스킵 (data_{did}): {lora_path}")
                    continue

                peft_model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=False)
                peft_model.eval()

                lora_results = []
                for qa in qa_pairs:
                    question = qa["question"]
                    ground_truth = qa["answer"]
                    
                    pred_text = predict(peft_model, tokenizer, generation_config, question, with_cot=WITH_COT, passages=None)
                    metrics = evaluate(pred_text, ground_truth)
                    
                    for k in lora_all_metrics: lora_all_metrics[k].append(metrics[k])
                    lora_results.append({"question": question, "ground_truth": ground_truth, "prediction": pred_text, **metrics})

                final_json_results.append({
                    "data_id": did,
                    "base_model_eval": base_results,
                    "lora_model_eval": lora_results
                })

                base_model = peft_model.unload()
                torch.cuda.empty_cache()
                gc.collect()

            # -------------------------------------------------
            # [Step 3] 파일별 결과 저장
            # -------------------------------------------------
            if not final_json_results:
                continue

            predict_file = os.path.join(dataset_out_dir, "predict_compare.json")
            with open(predict_file, "w", encoding="utf-8") as fout:
                json.dump(final_json_results, fout, indent=4, ensure_ascii=False)

            def calc_avg(metrics_dict):
                return {k: round(sum(float(x) for x in v)/len(v), 4) if len(v) > 0 else 0.0 for k, v in metrics_dict.items()}

            
            base_avg = calc_avg(base_all_metrics)
            lora_avg = calc_avg(lora_all_metrics)

            ret_str = f"===== Base Model Average ({prefix}) =====\n"
            for met in ["em", "f1", "prec", "recall"]: ret_str += f"{met}\t{base_avg[met]}\n"
            ret_str += f"\n===== LoRA Model Average ({prefix}) =====\n"
            for met in ["em", "f1", "prec", "recall"]: ret_str += f"{met}\t{lora_avg[met]}\n"

            with open(os.path.join(dataset_out_dir, "result.txt"), "w") as fout:
                fout.write(ret_str)

            print(f"[{file_name}] 평가 완료 (저장 경로: {dataset_out_dir})")

if __name__ == "__main__":
    main()