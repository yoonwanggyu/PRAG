import os
import json
import torch
from tqdm import tqdm
from peft import PeftModel
from utils import get_model

# ==========================================
# 1. 질문 리스트 세팅
# ==========================================
test_data_path = "test_data.json"

with open(test_data_path, "r", encoding="utf-8") as f:
    augmented_data_list = json.load(f)

golden_questions = []
counterfactual_questions = []
relevant_questions = []

for data in augmented_data_list:
    if data["pid"] == 0:
        golden_questions = data["golden_questions"]
        counterfactual_questions = data["counterfactual_questions"]
        # relevant_questions = data["relevant_questions"]
    # elif data["pid"] == 1:
    #     doc1_questions = data["test_questions"]
        
print(f"golden 질문 개수: {len(golden_questions)}개")
print(f"counterfactual 질문 개수: {len(counterfactual_questions)}개")
# print(f"relevant 질문 개수: {len(relevant_questions)}개")

# ==========================================
# 2. 추출 파이프라인
# ==========================================
def extract_hidden_representations(model_name, lora_path, questions_1, questions_2, output_path):
    print(f"### 1. Base Model & Tokenizer 로드 ({model_name}) ###")
    model, tokenizer, _ = get_model(model_name)
    
    print(f"### 2. LoRA 병합 ({lora_path}) ###")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval() # 추론 모드로 전환 (Dropout 등 비활성화)

    results = []

    print("### 3. Hidden Representation 추출 시작 ###")
    
    # 공통 추출 로직을 함수로 분리
    def process_questions(questions, doc_id):
        for q in tqdm(questions, desc=f"Extracting Doc {doc_id}"):
            # 질문을 토큰화하여 GPU로 올림
            inputs = tokenizer(q, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                # output_hidden_states=True를 주면 각 레이어별 결과를 모두 반환
                outputs = model(**inputs, output_hidden_states=True)
                
                # outputs.hidden_states는 튜플입니다. [-1]은 가장 마지막 레이어를 의미
                # shape: (batch_size, sequence_length, hidden_size)
                last_layer_hidden_states = outputs.hidden_states[-1]
                
                # 질문의 내용을 응축한 가장 "마지막 토큰"의 벡터만 추출
                # [0, -1, :] -> 첫 번째 배치(0), 마지막 토큰(-1), 전체 차원(:)
                last_token_repr = last_layer_hidden_states[0, -1, :].cpu().numpy().tolist()
                
            results.append({
                "question": q,
                "hidden_representation": last_token_repr,   # 질문 1개당 2048개의 숫자로 이루어진 벡터가 추출
                "doc_id": doc_id # 0 또는 1
            })

    process_questions(questions_1, doc_id=0)
    print(f"{questions_1} 질문 처리 완료")
    
    process_questions(questions_2, doc_id=1)
    print(f"{questions_2} 질문 처리 완료")

    print("### 4. 결과 저장 ###")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"총 {len(results)}개의 데이터가 {output_path}에 저장되었습니다!")

# ==========================================
# 3. 메인 실행부
# ==========================================
if __name__ == "__main__":
    MODEL_NAME = "llama3.2-1b-instruct"
    
    # 이전 단계에서 저장된 passage 0의 LoRA 경로
    LORA_PATH = "trained_LORA/passage_0_lora" 
    
    # 결과가 저장될 JSON 파일명
    OUTPUT_PATH = "output/doc0_golden_counterfactual.json"
    
    # 실행
    extract_hidden_representations(
        MODEL_NAME, 
        LORA_PATH, 
        golden_questions,
        counterfactual_questions, 
        OUTPUT_PATH
    )