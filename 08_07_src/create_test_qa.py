'''
GPT 이용해서 Test QA 만드는 코드
'''

import os
import json
import glob
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

DATASETS = ["2wikimultihopqa", "complexwebquestions", "hotpotqa", "popqa"]
MODEL_NAME = "gpt-5.6-sol" 

OUTPUT_DIR = os.path.join("re_src", "test_qa") 
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_json_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"JSON 파일 형식이 올바르지 않습니다: {filepath}")
        return None
    except Exception as e:
        print(f"에러가 발생했습니다 ({filepath}): {e}")
        return None

def generate_questions(passage_text, n_questions=10):
    prompt = f"""
        You are an expert reading comprehension AI. 
        Your task is to generate {n_questions} Question & Answer pairs based STRICTLY on the provided text.
        Do NOT use any external knowledge. If the answer cannot be found in the text, do not make the question.
        
        Text:
        {passage_text}
        
        Output MUST be a JSON object containing a list named 'qa_pairs'.
        Each item in the list must have exactly these three keys:
        - "question": The generated question.
        - "answer": A short, concise answer.
        - "full_answer": A complete sentence answering the question.
    """
 
    response = client.responses.create(
        model=MODEL_NAME,
        reasoning={"effort": "low"},
        text={"format": {"type": "json_object"}},
        input=[
            {"role": "system", "content": "You are a helpful assistant that generates Question and Answer pairs in JSON format."},
            {"role": "user", "content": prompt},
        ],
    )
 
    content = response.output_text
    
    try:
        qa_data = json.loads(content)
        return qa_data.get("qa_pairs", [])
    except json.JSONDecodeError:
        print("LLM 응답을 JSON으로 파싱할 수 없습니다.")
        print(f"원문:\n{content}")
        return []

def main():
    for dataset in DATASETS:
        dataset_dir = os.path.join("re_src", "data_aug", dataset, "llama3.2-1b-instruct")
        
        if not os.path.exists(dataset_dir):
            print(f"경로가 존재하지 않습니다. 스킵합니다: {dataset_dir}")
            continue

        json_files = glob.glob(os.path.join(dataset_dir, "*.json"))
        
        for file_path in json_files:
            file_name = os.path.basename(file_path) 
            prefix = file_name.replace(".json", "")
            
            print(f"\n처리 시작: [{dataset}] 데이터셋의 [{file_name}] 파일")
            
            items = load_json_file(file_path)
            if not items:
                continue
 
            results = []
            for item in items:
                passages = item.get("passages", [])
                if not passages:
                    continue
            
                combined_text = "\n".join(passages)
 
                try:
                    questions_data = generate_questions(combined_text, n_questions=10)
                except Exception as e:
                    print(f"GPT 호출 중 에러: {e}")
                    continue
 
                item[""]
                results.append(item)
 
            out_path = os.path.join(OUTPUT_DIR, f"{dataset}_{prefix}_questions.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
 
            print(f"처리 완료: {len(results)}개 항목, 저장 위치 {out_path}")

if __name__ == "__main__":
    main()
