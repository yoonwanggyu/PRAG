import os
import json
import torch
from tqdm import tqdm
from peft import PeftModel
from utils import get_model

# ==========================================
# 1. 질문 리스트 세팅
# ==========================================
test_data_path = "test_questions.json"

with open(test_data_path, "r", encoding="utf-8") as f:
    test_questions_list = json.load(f)

test_questions = []

for data in test_questions_list:
    test_questions.append(data['question'])
        
print(f"test 질문 개수: {len(test_questions)}개")

# ==========================================
# 2. 추출 파이프라인 (LoRA A,B 한번씩 장착해서 각각의 hidden representation 추출)
# ==========================================
def extract_hidden_representations(model_name, lora_paths, questions, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"### 1. Base Model & Tokenizer 로드 ({model_name}) ###")
    base_model, tokenizer, _ = get_model(model_name)

    # output 폴더 생성
    os.makedirs(output_folder,exist_ok=True)
    
    print(f"### 2. LoRA 병합 ({lora_paths}) ###")
    for i,path in enumerate(lora_paths):

        print(f"\n--- [Doc {i}] LoRA 로드 중: {path} ---")

        model = PeftModel.from_pretrained(base_model, path)
        model.eval() # 추론 모드로 전환 (Dropout 등 비활성화)

        results = []

        print("### 3. Hidden Representation 추출 시작 ###")
        for q in tqdm(questions, desc=f"Extracting Doc {i}"):
            # 질문을 토큰화하여 GPU로 올림
            inputs = tokenizer(q, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                # output_hidden_states=True를 주면 각 레이어별 결과를 모두 반환
                outputs = model(**inputs, output_hidden_states=True)
                
                # outputs.hidden_states는 튜플입니다. [-1]은 가장 마지막 레이어를 의미
                # shape: (batch_size, sequence_length, hidden_size)
                # 트랜스포머의 모든 디코더 블록(Self-Attention, FFN, 그리고 내부의 Add & Norm)을 전부 통과한 후, 
                #  마지막 LM Head(단어 예측을 위한 Linear 레이어)로 들어가기 직전의 최종 벡터
                # last_layer_hidden_states = outputs.hidden_states[-1]
                last_layer_hidden_states = outputs.hidden_states[-2]
                
                # ---방법 1. last token
                # 질문의 내용을 응축한 가장 "마지막 토큰"의 벡터만 추출
                # [0, -1, :] -> 첫 번째 배치(0), 마지막 토큰(-1), 전체 차원(:)
                last_token_repr = last_layer_hidden_states[0, -1, :].cpu().numpy().tolist()
                
                # --방법 2. Mean Pooling
                # sentence_vector = last_layer_hidden_states[0].mean(dim=0)
                # mean_pooled_repr = sentence_vector.cpu().numpy().tolist()

            results.append({
                "question": q,
                "hidden_representation": last_token_repr,   # 질문 1개당 2048개의 숫자로 이루어진 벡터가 추출
                "doc_id": i
            })
    
        # 이렇게 하면 다음 루프(i+1)에서 PeftModel.from_pretrained(base_model, path)를 할 때
        # 완벽하게 깨끗한 상태에서 새 LoRA를 씌울 수 있습니다.
        base_model = model.unload()

        # doc별 파일 저장
        output_path = os.path.join(output_folder, f"doc{i}_hidden_representation.json")
        # output_path = os.path.join(output_folder, f"base_weight_hidden_representation.json")

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
    LORA_PATHs = ["trained_LORA/passage_0_lora", "trained_LORA/passage_1_lora"]
    # LORA_PATHs = ["trained_LORA/base_weight"]
    
    # 실행
    extract_hidden_representations(
        MODEL_NAME, 
        LORA_PATHs, 
        test_questions,
        output_folder="output/pre"
    )