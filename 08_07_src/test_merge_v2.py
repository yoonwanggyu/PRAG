'''
같은 데이터셋에서 서로 다른 문서 LoRA를 병합해 테스트하는 코드
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
    OUTPUT_ROOT_DIR = os.path.join("re_src", "merge_v2_output_eval")
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

    QA_KEY = "generated_questions" 

    generation_config = dict(
        num_beams=1, 
        do_sample=False,
        max_new_tokens=50,
        return_dict_in_generate=True,
    )

    for dataset in DATASETS:
        print(f"\n{'='*60}\n [Dataset: {dataset}] 동일 데이터셋 내 4개 문서 병합 및 평가 시작\n{'='*60}")

        if dataset in ["2wikimultihopqa", "hotpotqa"]:
            cot_name = "cot"
            prefix = "bridge_comparison" if dataset == "2wikimultihopqa" else "bridge"
        else:
            cot_name = "direct"
            prefix = "total"

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
        
        if generation_config.get("pad_token_id") is None:
            generation_config["pad_token_id"] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        print("동일 데이터셋 내의 문서 4개(data_0 ~ data_3) LoRA 병합")
        peft_model = None
        loaded_adapters = []
        
        for doc_id in range(3):
            lora_path = os.path.join(
                "offline", "llama3.2-1b-instruct", "rank=2_alpha=32", 
                dataset, f"lr=0.0003_epoch=1_{cot_name}", "aug_model=llama3.2-1b-instruct", 
                prefix, "merged_passage", f"data_{doc_id}"
            )
            
            if os.path.exists(lora_path):
                print(f"[{dataset}] 문서 {doc_id} LoRA 로드 중... 경로: {lora_path}")
                if peft_model is None:
                    peft_model = PeftModel.from_pretrained(base_model, lora_path, adapter_name=str(doc_id), is_trainable=False)
                else:
                    peft_model.load_adapter(lora_path, adapter_name=str(doc_id), is_trainable=False)
                loaded_adapters.append(str(doc_id))
            else:
                print(f"경고: [{dataset}] 문서 {doc_id} LoRA 경로를 찾을 수 없습니다. ({lora_path})")

        if not loaded_adapters:
            print(f"[{dataset}] 로드된 어댑터가 없어 평가를 건너뜁니다.")
            del base_model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            continue

        print("\n로드된 어댑터들을 'cat' 방식으로 병합합니다...")
        peft_model.add_weighted_adapter(
            adapters = loaded_adapters, 
            # weights = [1.0 / len(loaded_adapters)] * len(loaded_adapters),
            weights = [1.0] * len(loaded_adapters),
            adapter_name = "merge", 
            combination_type = "cat",
        )
        
        peft_model.set_adapter("merge")
        merged_model = peft_model.merge_and_unload()
        merged_model.eval()

        print("\n문서 3개 병합 완료. 첫 번째 문서(data_0) QA로 테스트를 진행합니다.")

        search_pattern = os.path.join(QA_DIR, f"{dataset}_*questions.json")
        json_files = glob.glob(search_pattern)

        if not json_files:
            print(f"파일 없음 스킵: {search_pattern}")
            del merged_model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            continue

        for file_path in json_files:
            qa_data_list = load_json_file(file_path)
            if not qa_data_list: break

            # 타겟 문서 인덱스를 0 (첫 번째 문서)으로 고정
            target_data_id = 0 
            if len(qa_data_list) <= target_data_id:
                print(f"경고: 파일에 data_{target_data_id}가 존재하지 않습니다.")
                continue

            target_item = qa_data_list[target_data_id]
            qa_pairs = target_item.get(QA_KEY, [])
            
            if not qa_pairs:
                print(f"경고: data_{target_data_id}에 '{QA_KEY}' 키가 없거나 비어있습니다.")
                continue

            dataset_out_dir = os.path.join(OUTPUT_ROOT_DIR, dataset, prefix)
            os.makedirs(dataset_out_dir, exist_ok=True)

            final_json_results = []
            lora_all_metrics = {"em": [], "f1": [], "prec": [], "recall": []}
            lora_results = []

            for qa in tqdm(qa_pairs, desc=f"Evaluating {dataset} (Target: data_0)"):
                question = qa["question"]
                ground_truth = qa["answer"]
                
                pred_text = predict(merged_model, tokenizer, generation_config, question, with_cot=WITH_COT, passages=None)
                metrics = evaluate(pred_text, ground_truth)
                
                for k in lora_all_metrics: lora_all_metrics[k].append(metrics[k])
                lora_results.append({"question": question, "ground_truth": ground_truth, "prediction": pred_text, **metrics})

            final_json_results.append({
                "merged_docs": loaded_adapters,           # 병합에 사용된 문서 인덱스 목록
                "target_test_data_id": target_data_id,    # 평가에 사용된 타겟 문서(0)
                "all_merged_model_eval": lora_results
            })

            print("결과 저장 중")
            predict_file = os.path.join(dataset_out_dir, "predict_same_dataset_merged.json")
            with open(predict_file, "w", encoding="utf-8") as fout:
                json.dump(final_json_results, fout, indent=4, ensure_ascii=False)

            def calc_avg(metrics_dict):
                return {k: round(sum(float(x) for x in v)/len(v), 4) if len(v) > 0 else 0.0 for k, v in metrics_dict.items()}

            lora_avg = calc_avg(lora_all_metrics)

            ret_str = f"===== Merged LoRA (4 Docs) Evaluation on data_0 ({prefix}) =====\n"
            for met in ["em", "f1", "prec", "recall"]: ret_str += f"{met}\t{lora_avg[met]}\n"

            with open(os.path.join(dataset_out_dir, "result.txt"), "w") as fout:
                fout.write(ret_str)

            print(f"\n[{os.path.basename(file_path)}] 평가 완료 (저장 경로: {dataset_out_dir})")

        print(f"[{dataset}] 평가가 완료되어 GPU 메모리를 비웁니다.")
        del merged_model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()