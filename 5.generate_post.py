import torch
from transformers import pipeline

model_path = "output/checkpoint-5600"
model_name = "gpt2"

# 檢查是否支援 CUDA
cuda_available = torch.cuda.is_available()
device = 0 if cuda_available else -1
print("===偵測設備是否有顯卡===")
print("使用 GPU" if cuda_available else "使用 CPU")

generator = pipeline("text-generation", model=model_path, tokenizer=model_name, device=device)

def generate_text(prompt, max_length=150):
    return generator(prompt, max_length=max_length, num_return_sequences=1)[0]["generated_text"]

prompts = [
    "今天天氣很好",
    "和冠傑他們去爬山",
    "認識了叫做EVA的新朋友",
    "吃了好吃的餐廳名叫皇椰雞"
]

generated_article = ""

for prompt in prompts:
    output_text = generate_text(prompt)
    generated_article += output_text + " "

print("Generated article:")
print(generated_article)