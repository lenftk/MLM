import torch
from model import Config, MaskedLanguageModel
from tokenizers import Tokenizer

# 모델 초기화
config = Config(vocab_size=36000, hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024, max_position_embeddings=128)
model = MaskedLanguageModel(config)
model.load_state_dict(torch.load("C:/Users/USER/Desktop/Project_LLM/model/model.pt"))
model.eval()

# 토크나이저 초기화
tokenizer = Tokenizer.from_file("tokenizer.json")

# 사용자 입력
user_input = "I saw a apple on the chire"

# 입력 전처리
input_ids = tokenizer.encode(user_input).ids
input_tensor = torch.tensor(input_ids).unsqueeze(0)  # 배치 차원 추가

# 모델 호출
with torch.no_grad():
    output = model(input_tensor)

# 출력 후처리
output_ids = torch.argmax(output[0], dim=-1).tolist()
output_text = tokenizer.decode(output_ids)

print("모델 응답:", output_text)
