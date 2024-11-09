from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. StarCoder2 모델과 토크나이저 설정
def load_starcoder2_model():
    model_name = "bigcode/starcoder2-15b"  # StarCoder2 모델 이름
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

# 2. 코드 입력을 모델에 전달하여 생성 결과 반환
def generate_code_with_starcoder2(model, tokenizer, input_code):
    # 입력 코드를 토큰화
    input_ids = tokenizer(input_code, return_tensors="pt").input_ids

    # 모델을 사용하여 코드 생성 (GPU 사용 가능 시 GPU로 이동)
    if torch.cuda.is_available():
        model = model.to("cuda")
        input_ids = input_ids.to("cuda")
    
    # 모델로부터 결과 생성
    output_ids = model.generate(input_ids, max_length=256, num_return_sequences=1)
    generated_code = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_code

# 3. 예제 코드 실행
if __name__ == "__main__":
    # StarCoder2 모델과 토크나이저 불러오기
    model, tokenizer = load_starcoder2_model()
    
    # 사용자 코드 입력 (예시 코드)
    input_code = "def hello_world():\n    print('Hello, world!')"
    
    # 코드 생성 결과 얻기
    generated_code = generate_code_with_starcoder2(model, tokenizer, input_code)
    
    # 결과 출력
    print("Generated Code:\n", generated_code)
