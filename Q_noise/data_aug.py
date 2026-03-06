import json
from tqdm import tqdm
from utils import get_model, model_generate 

# 1. Rewrite 함수
def get_rewrite(passage, model_name, model=None, tokenizer=None, generation_config=None):
    rewrite_prompt = "Rewrite the following passage. While keeping the entities, proper nouns, and key details such as names, locations, and terminology intact, create a new version of the text that expresses the same ideas in a different way. Make sure the revised passage is distinct from the original one, but preserves the core meaning and relevant information.\n{passage}"
    return model_generate(rewrite_prompt.format(passage=passage), model, tokenizer, generation_config)

# 2. QA 생성 프롬프트
qa_prompt_template = "I will provide a passage of text, and you need to generate three different questions based on the content of this passage. Each question should be answerable using the information provided in the passage. Additionally, please provide an appropriate answer for each question derived from the passage.\n\
You need to generate the question and answer in the following format:\n\
[\n\
    {{\n\
        \"question\": \"What is the capital of France?\",\n\
        \"answer\": \"Paris\"\n\
        \"full_answer\": \"The capital of France is Paris.\"\n\
    }}, \n\
]\n\n\
This list should have at least three elements. You only need to output this list in the above format.\n\
Passage:\n\
{passage}"

# 3. QA 결과 포맷 검증 함수
def fix_qa(qa):
    if isinstance(qa, list):
        if len(qa) >= 3:
            qa = qa[:3]
            for data in qa:
                if "question" not in data or "answer" not in data or "full_answer" not in data:
                    return False, qa
                if isinstance(data["answer"], list):
                    data["answer"] = ", ".join(data["answer"])
                if isinstance(data["answer"], int):
                    data["answer"] = str(data["answer"])
                if data["answer"] is None:
                    data["answer"] = "Unknown"
            return True, qa
    return False, qa

# 4. QA 생성 함수 (재시도 및 JSON 파싱 로직 포함)
def get_qa(passage, model_name, model=None, tokenizer=None, generation_config=None):
    def fix_json(output):
        if model_name == "llama3.2-1b-instruct":
            output = output[output.find("["):]
            if output.endswith(","):
                output = output[:-1]
            if not output.endswith("]"):
                output += "]"
        elif model_name == "llama3-8b-instruct":
            if "[" in output:
                output = output[output.find("["):] 
            if "]" in output:
                output = output[:output.find("]")+1]
        return output

    try_times = 100
    prompt = qa_prompt_template.format(passage=passage)
    output = None
    while try_times:
        output = model_generate(prompt, model, tokenizer, generation_config)
        output = fix_json(output)
        try:
            qa = json.loads(output)
            ret, qa = fix_qa(qa)
            if ret:
                return qa
        except:
            try_times -= 1
    return output

def process_my_passages(my_passages_list, model_name, output_file_path):
    print(f"### Loading Model: {model_name} ###")
    model, tokenizer, _ = get_model(model_name)
    
    generation_config = dict(
        max_new_tokens=512,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        temperature=0.7,
        top_k=50,
    )

    results = []
    print("### Starting Rewrite & QA Generation ###")
    
    for pid, psg in enumerate(tqdm(my_passages_list)):
        # 1. Rewrite 
        rewritten_text = get_rewrite(psg, model_name, model, tokenizer, generation_config)
        
        # 2. QA 
        qa_pairs = get_qa(psg, model_name, model, tokenizer, generation_config)
        
        # 3. QA 검증 및 저장
        is_valid, final_qa = fix_qa(qa_pairs)
        
        val = {
            "pid": pid,
            "original_passage": psg,
            f"{model_name}_rewrite": rewritten_text,
            f"{model_name}_qa": final_qa if is_valid else "QA Generation Failed"
        }
        results.append(val)

    # 결과 JSON 저장
    with open(output_file_path, "w", encoding='utf-8') as fout:
        json.dump(results, fout, indent=4, ensure_ascii=False)
        
    print(f"### Process Completed. Saved to {output_file_path} ###")

if __name__ == "__main__":
    # 1. MD 파일 경로 설정 
    md_file_path = "Q_noise/data.md" 
    
    my_data_list = []
    
    # 2. 파일을 읽어서 # 기준으로 나누고 텍스트만 추출하기
    with open(md_file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # '# '로 시작하는 블록들로 쪼갭니다.
    blocks = content.split('# ')
    
    for block in blocks:
        if not block.strip():
            continue # 빈 블록은 무시
            
        # 줄바꿈을 기준으로 첫 줄(제목)과 나머지(내용) 분리
        lines = block.strip().split('\n')
        passage_text = '\n'.join(lines[1:]).strip()
        
        # 양 끝에 있는 큰따옴표(") 제거
        if passage_text.startswith('"') and passage_text.endswith('"'):
            passage_text = passage_text[1:-1]
            
        # \u00f1 같은 이스케이프 문자를 실제 문자로 변환 (필요한 경우)
        try:
            passage_text = passage_text.encode('utf-8').decode('unicode_escape')
        except:
            pass
            
        if passage_text:
            my_data_list.append(passage_text)
            
    print(f"총 {len(my_data_list)}개의 passage를 성공적으로 불러왔습니다.")
    
    # 3. 사용할 모델 이름 설정
    MODEL_NAME = "llama3.2-1b-instruct"
    
    # 4. 결과 저장 경로
    OUTPUT_FILE = "Q_noise/data_aug_result.json"
    
    # 5. 실행!
    process_my_passages(
        my_passages_list=my_data_list, 
        model_name=MODEL_NAME, 
        output_file_path=OUTPUT_FILE
    )