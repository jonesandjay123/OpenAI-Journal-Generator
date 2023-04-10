from transformers import pipeline

model_path = "output"
generator = pipeline("text-generation", model=model_path, tokenizer=model_name, device=0)

def generate_text(prompt, max_length=150):
    return generator(prompt, max_length=max_length, num_return_sequences=1)[0]["generated_text"]

prompt = "今天天氣很好"  # 您想要提供的主題
output_text = generate_text(prompt)
print(output_text)
