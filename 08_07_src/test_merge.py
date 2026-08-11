'''
서로 다른 데이터셋 4개 각각에서 문서 LoRA 총 4개 병합해서 test하는 코드
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
    OUTPUT_ROOT_DIR = os.path.join("re_src", "dif_merge_output_eval_rank8")
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

    print("\n" + "="*50)
    print("4개 데이터셋 LoRA 어댑터 로드 및 'cat' 병합 시작")
    print("="*50)
    
    peft_model = None
    loaded_adapters = []

    for i, dataset in enumerate(DATASETS):
        if dataset in ["2wikimultihopqa", "hotpotqa"]:
            cot_name = "cot"
            if dataset == "2wikimultihopqa":
                prefix = "bridge_comparison"
            else:
                prefix = "bridge"
        else:
            cot_name = "direct"
            prefix = "total"

        lora_path = os.path.join(
                    "offline", "llama3.2-1b-instruct", "rank=8_alpha=32", 
                    dataset, f"lr=0.0003_epoch=3_{cot_name}", "aug_model=llama3.2-1b-instruct", 
                    prefix, "merged_passage", f"data_0", 
                )
        
        if os.path.exists(lora_path):
            print(f"[{dataset}] LoRA 로드 중... (adapter_name: {i}) 경로: {lora_path}")
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(base_model, lora_path, adapter_name=str(i), is_trainable=False)
            else:
                peft_model.load_adapter(lora_path, adapter_name=str(i), is_trainable=False)
            loaded_adapters.append(str(i))
        else:
            print(f"경고: [{dataset}] LoRA 경로를 찾을 수 없어 건너뜁니다. ({lora_path})")

    if not loaded_adapters:
        print("로드된 LoRA 어댑터가 없어 종료합니다.")
        return

    print("\n로드된 어댑터들을 'cat' 방식으로 병합합니다...")
    peft_model.add_weighted_adapter(
        adapters = [str(i) for i in range(len(loaded_adapters))], 
        weights = [1] * len(loaded_adapters),
        # weights = [1.0 / len(loaded_adapters)] * len(loaded_adapters),
        adapter_name = "merge", 
        combination_type = "cat",
    )
    
    peft_model.set_adapter("merge")
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()
    
    print("\n모든 LoRA 병합 완료. 단일 모델로 평가를 시작합니다.")

    for dataset in DATASETS:
        print(f"\n{'='*50}\n Dataset: {dataset} 평가 시작\n{'='*50}")

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
            lora_all_metrics = {"em": [], "f1": [], "prec": [], "recall": []}

            for did, item in enumerate(tqdm(qa_data_list, desc=f"Evaluating {prefix}")):
                qa_pairs = item.get(QA_KEY, [])
                if not qa_pairs: 
                    continue

                lora_results = []
                for qa in qa_pairs:
                    question = qa["question"]
                    ground_truth = qa["answer"]
                    
                    pred_text = predict(merged_model, tokenizer, generation_config, question, with_cot=WITH_COT, passages=None)
                    metrics = evaluate(pred_text, ground_truth)
                    
                    for k in lora_all_metrics: lora_all_metrics[k].append(metrics[k])
                    lora_results.append({"question": question, "ground_truth": ground_truth, "prediction": pred_text, **metrics})

                final_json_results.append({
                    "data_id": did,
                    "all_merged_model_eval": lora_results
                })

            if not final_json_results:
                continue

            predict_file = os.path.join(dataset_out_dir, "predict_cat_merged.json")
            with open(predict_file, "w", encoding="utf-8") as fout:
                json.dump(final_json_results, fout, indent=4, ensure_ascii=False)

            def calc_avg(metrics_dict):
                return {k: round(sum(float(x) for x in v)/len(v), 4) if len(v) > 0 else 0.0 for k, v in metrics_dict.items()}

            lora_avg = calc_avg(lora_all_metrics)

            ret_str = f"===== Cat Merged LoRA Model Average ({prefix}) =====\n"
            for met in ["em", "f1", "prec", "recall"]: ret_str += f"{met}\t{lora_avg[met]}\n"

            with open(os.path.join(dataset_out_dir, "result.txt"), "w") as fout:
                fout.write(ret_str)

            print(f"[{file_name}] 평가 완료 (저장 경로: {dataset_out_dir})")

if __name__ == "__main__":
    main()