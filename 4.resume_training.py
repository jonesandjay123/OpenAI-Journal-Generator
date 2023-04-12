import os
import json
import glob
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

output_dir = "output"
latest_checkpoint = max(glob.glob(os.path.join(output_dir, "checkpoint-*")), key=os.path.getmtime)
latest_step = int(re.search(r"checkpoint-(\d+)", latest_checkpoint).group(1))
model_path = latest_checkpoint
save_steps = 1600 - (latest_step % 1600)

# 載入 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 載入模型
model = GPT2LMHeadModel.from_pretrained(model_path)

# 載入訓練集與驗證集
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="train.txt",
    block_size=128
)
val_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="val.txt",
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 讀取 trainer_state.json 文件
with open(os.path.join(model_path, "trainer_state.json"), "r") as f:
    trainer_state_dict = json.load(f)

# 計算剩餘的 epoch 數
num_epochs = 5 - trainer_state_dict["epoch"]

latest_step = trainer_state_dict["global_step"]
next_save_step = 1600 - (latest_step % 1600)

# 更新 TrainingArguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_steps=200,
    save_steps=next_save_step,
    warmup_steps=50,
    logging_steps=100,
    prediction_loss_only=True,
    resume_from_checkpoint=model_path,
    num_train_epochs=num_epochs,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    save_total_limit=2,
    report_to="none",
    evaluation_strategy="steps",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
trainer.save_model(model_path)
