import os
import gc
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import (
    DataCollatorForCompletionOnlyLM,
    SFTConfig, 
    SFTTrainer
)
from pathlib import Path
import pandas as pd
from string import ascii_letters

import random
import torch
import numpy as np

# Seed Everything
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Model to Fine-Tune
MODEL_NAME = os.environ['FT_MODEL']

# Datasets
train_paths = [str(x) for x in Path('data/').glob('*train.json')]

for train_path in train_paths:
    print(f"Fine-tune {MODEL_NAME} on {train_path}.")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ["HF_AUTH_TOKEN"])
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype = torch.bfloat16,
        device_map = 'auto',
        token=os.environ["HF_AUTH_TOKEN"],
    )
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    def format_example(row):
        prompt = row['input'] + '\n\n### Options:\n'+'\n'.join([f'{ascii_letters[i]} : {row["answer_choices"][i]}' for i in range(len(row['answer_choices']))])
        answer = "### Correct Answer:\n" + ascii_letters[row['label']]
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)
    
    def count_tokens(row):
        return len(tokenizer(row["text"], add_special_tokens=True, return_attention_mask=False)['input_ids'])
    
    df = pd.read_json(train_path, lines=True)
    df['text'] = df.apply(format_example, axis=1)
    df['num_tokens'] = df.apply(count_tokens, axis=1)
    df = df[df.num_tokens < 512]
    df.to_json("train.json", orient='records', lines=True)
    dataset = load_dataset("json", data_files={"train": "train.json"})
    response_template = "### Correct Answer:\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules='all-linear',
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    
    OUT_DIR = f'{os.environ["OUT_DIR"]}{MODEL_NAME}/{train_path.split("/")[-1].split(".")[0]}'
    
    sft_config = SFTConfig(
        do_train=True,
        output_dir=OUT_DIR,
        dataset_text_field="text",
        max_seq_length=512,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        save_strategy='epoch',
        learning_rate=1e-4,
        lr_scheduler_type='constant',
        bf16=True,
        seed=42,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False
        },
        torch_empty_cache_steps=100,
    )
    
    trainer = SFTTrainer(
        model = model,
        args=sft_config,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        data_collator=collator,
    )
    
    trainer.train()
    del model
    del trainer
    del dataset
    gc.collect()
    torch.cuda.empty_cache()
    
    