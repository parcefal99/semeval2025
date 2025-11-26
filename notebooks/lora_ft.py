import os, random, functools, csv
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from datasets import Dataset, DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)

MODEL_PATH = "/l/users/abzal.nurgazy/models/llama32_3b" 
DATASET_PATH = "/home/abzal.nurgazy/semeval2025/dataset"
TRAIN_PATH = "esp_train.csv"
DEV_PATH   = "esp_dev.csv"

print(f"Model Path: {MODEL_PATH}, Train Path {TRAIN_PATH}")


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, csv_path, tokenizer, max_length=512):
        self.data = pd.read_csv(os.path.join(dataset_path, csv_path), encoding='utf-8')
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_cols = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
        self.num_labels = len(self.label_cols)
        self.label_weights = 1 - self.data[self.label_cols].sum(axis=0) / self.data[self.label_cols].sum().sum()

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text']
        labels = row[self.label_cols].values.astype(np.float32)

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(labels, dtype=torch.float32)
        return item

train_dataset = EmotionDataset(DATASET_PATH, TRAIN_PATH, tokenizer)
dev_dataset   = EmotionDataset(DATASET_PATH, DEV_PATH, tokenizer)


def collate_fn(batch, tokenizer):
    keys = ['input_ids', 'attention_mask', 'labels']
    d = {k: [ex[k] for ex in batch] for k in keys}
    d['input_ids'] = torch.nn.utils.rnn.pad_sequence(d['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id)
    d['attention_mask'] = torch.nn.utils.rnn.pad_sequence(d['attention_mask'], batch_first=True, padding_value=0)
    d['labels'] = torch.stack(d['labels'])
    return d


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_type=torch.bfloat16, 
    bnb_4bit_use_double_quant=True
)

config = AutoConfig.from_pretrained(
    MODEL_PATH,
    num_labels=train_dataset.num_labels,
    problem_type="multi_label_classification",
)


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    config=config,
    quantization_config=quantization_config,
    device_map="auto",
    attn_implementation="eager" 
)


model.config.pad_token_id = tokenizer.pad_token_id

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)


lora_config = LoraConfig(
    r=8, 
    lora_alpha=32,
    target_modules=['q_proj','v_proj','o_proj'], 
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
    modules_to_save=["score"],   
)

model = get_peft_model(model, lora_config)

def compute_metrics(p):
    logits = p.predictions

    if isinstance(logits, tuple):
        logits = logits[0]
        
    labels = p.label_ids

    preds = (logits > 0).astype(int)
    return {
        'f1_micro':    f1_score(labels, preds, average='micro',    zero_division=0),
        'f1_macro':    f1_score(labels, preds, average='macro',    zero_division=0),
        'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
    }


class MultilabelTrainer(Trainer):
    def __init__(self, label_weights, **kwargs):
        super().__init__(**kwargs)
        self.label_weights = label_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  

        pos_weight = self.label_weights.to(logits.device)

        loss = F.binary_cross_entropy_with_logits(
            logits.float(), 
            labels.to(torch.float32),
            pos_weight=pos_weight,
            reduction="mean"
        )
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir='multilabel_gemma_classification',
    overwrite_output_dir=True,   
    
    learning_rate=5e-5,          
    max_grad_norm=0.3,           
    weight_decay=0.01,

    
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    report_to="none",
    
    fp16=False,
    bf16=True,       
    logging_steps=10,
)

trainer = MultilabelTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,           
    tokenizer=tokenizer,
    data_collator=functools.partial(collate_fn, tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    label_weights=torch.tensor(train_dataset.label_weights.values,
                               device=model.device, dtype=torch.float32),
)

print_trainable_parameters(trainer.model)

torch.cuda.empty_cache()

trainer.train()

peft_model_id = 'multilabel_llama32_3b_lora_esp'
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
print(f"Model saved to {peft_model_id}")