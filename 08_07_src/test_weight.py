import os
import re
import json
import torch
import string
import gc
import glob
from tqdm import tqdm
import numpy as np
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
    MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    
    # [수정 필요] 학습하셨던 최신 하이퍼파라미터에 맞게 경로 확인
    RANK = 2
    ALPHA = 32
    EPOCH = 1
    
    # 1. 평가할 타겟 데이터셋 (PopQA)
    TARGET_DATASET = "popqa"
    TARGET_PREFIX = "total"
    TARGET_QA_FILE = f"re_src/test_qa/{TARGET_DATASET}_{TARGET_PREFIX}_questions.json"
    TARGET_WITH_COT = False 
    
    # LoRA 경로 설정
    HOTPOTQA_LORA_PATH = os.path.join(
        "offline", "llama3.2-1b-instruct", f"rank={RANK}_alpha={ALPHA}", 
        "hotpotqa", f"lr=0.0003_epoch={EPOCH}_cot", "aug_model=llama3.2-1b-instruct", 
        "bridge", "merged_passage", "data_0"
    )
    
    POPQA_LORA_PATH = os.path.join(
        "offline", "llama3.2-1b-instruct", f"rank={RANK}_alpha={ALPHA}", 
        "popqa", f"lr=0.0003_epoch={EPOCH}_direct", "aug_model=llama3.2-1b-instruct", 
        "total", "merged_passage", "data_0"
    )

    OUTPUT_DIR = os.path.join("re_src", "interpolation_eval")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generation_config = dict(
        num_beams=1, 
        do_sample=False,
        max_new_tokens=50,
        return_dict_in_generate=True,
    )

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
    generation_config["pad_token_id"] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    print("\n어댑터(LoRA) 2개 로드 중...")
    peft_model = PeftModel.from_pretrained(base_model, HOTPOTQA_LORA_PATH, adapter_name="hotpot", is_trainable=False)
    peft_model.load_adapter(POPQA_LORA_PATH, adapter_name="popqa", is_trainable=False)

    # 타겟 QA 데이터 로드 (data_0)
    qa_data_list = load_json_file(TARGET_QA_FILE)
    if not qa_data_list:
        print("타겟 QA 파일을 찾을 수 없습니다.")
        return
        
    target_qa_pairs = qa_data_list[0].get("generated_questions", [])
    if not target_qa_pairs:
        print("QA 데이터가 비어있습니다.")
        return

    # 가중치 보간 범위 설정 (HotpotQA 비중: 1.0 -> 0.0)
    alphas = np.linspace(1.0, 0.0, 11) # [1.0, 0.9, 0.8, ..., 0.0]
    
    summary_results = []
    
    print("\n" + "="*60)
    print(" 가중치 보간(Weight Interpolation) 실험 시작")
    print("="*60)

    for w_target in alphas:
        w_target = round(w_target, 2)
        w_noise = round(1.0 - w_target, 2)
        
        merge_name = f"merge_t{w_target}_n{w_noise}".replace(".", "_")
        print(f"\n[실험 진행 중] PopQA Weight: {w_target:.1f} | HotpotQA Weight: {w_noise:.1f}")

        # 어댑터 가중치 병합
        peft_model.add_weighted_adapter(
            adapters=["popqa", "hotpot"], 
            weights=[w_target, w_noise],
            adapter_name=merge_name, 
            combination_type="cat" 
        )
        peft_model.set_adapter(merge_name)
        
        # 임시 병합 모델 평가 모드
        peft_model.eval()

        lora_all_metrics = {"em": [], "f1": [], "prec": [], "recall": []}
        
        for qa in tqdm(target_qa_pairs, desc=f"Evaluating weights ({w_target}:{w_noise})", leave=False):
            question = qa["question"]
            ground_truth = qa["answer"]
            
            pred_text = predict(peft_model, tokenizer, generation_config, question, with_cot=TARGET_WITH_COT, passages=None)
            metrics = evaluate(pred_text, ground_truth)
            
            for k in lora_all_metrics: 
                lora_all_metrics[k].append(metrics[k])

        # 평균 계산
        avg_metrics = {k: round(sum(float(x) for x in v)/len(v), 4) if len(v) > 0 else 0.0 for k, v in lora_all_metrics.items()}
        
        # 콘솔 출력
        print(f"-> 결과: EM: {avg_metrics['em']:.4f} | F1: {avg_metrics['f1']:.4f}")
        
        summary_results.append({
            "w_hotpot": w_target,
            "w_popqa": w_noise,
            "metrics": avg_metrics
        })

        # --- 이 부분 추가 (메모리 확보를 위해 사용 끝난 어댑터 삭제) ---
        peft_model.delete_adapter(merge_name)

    # 결과 저장
    out_json_path = os.path.join(OUTPUT_DIR, "weight_interpolation_results.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=4)

    out_txt_path = os.path.join(OUTPUT_DIR, "interpolation_summary.txt")
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("===== Weight Interpolation Summary =====\n")
        f.write("w_hotpot\tw_popqa\tEM\tF1\tPrecision\tRecall\n")
        for res in summary_results:
            m = res["metrics"]
            f.write(f"{res['w_hotpot']:.1f}\t{res['w_popqa']:.1f}\t{m['em']}\t{m['f1']}\t{m['prec']}\t{m['recall']}\n")
            
    print(f"\n모든 평가 완료! 결과가 {OUTPUT_DIR} 에 저장되었습니다.")

if __name__ == "__main__":
    main()
