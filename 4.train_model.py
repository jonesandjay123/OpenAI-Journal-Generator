import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name) // 首次運行時使用這一行來創建檔案，之後則使用下面的版本延續訓練

# 將已經訓練過的模型載入，您可以通過指定已保存模型的目錄路徑
model_path = os.path.abspath("output")
model = GPT2LMHeadModel.from_pretrained(model_path)

def load_dataset(train_path, val_path, tokenizer):
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128
    )
    val_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=val_path,
        block_size=128
    )
    return train_dataset, val_dataset

train_dataset, val_dataset = load_dataset("train.txt", "val.txt", tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_steps=200,
    save_steps=800,
    warmup_steps=50,
    logging_steps=100,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
trainer.save_model("output")
