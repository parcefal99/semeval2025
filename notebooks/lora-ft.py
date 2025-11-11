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

MODEL_PATH = "/l/users/abzal.nurgazy/models/gemma4b"
DATASET_PATH = "/home/abzal.nurgazy/semeval2025/dataset"
TRAIN_PATH = "eng_train.csv"
# TEST_PATH  = "eng_test.csv"
DEV_PATH   = "eng_dev.csv"


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def print_trainable_parameters(model):
    """
  printing the number of trainable paramters in the model
  """
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
        self.label_cols = ["anger", "fear", "joy", "sadness", "surprise"]
        self.num_labels = len(self.label_cols)
        # per-class pos_weight (higher for rarer classes)
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
test_dataset  = EmotionDataset(DATASET_PATH, TEST_PATH, tokenizer)

# --- Collator ---
def collate_fn(batch, tokenizer):
    keys = ['input_ids', 'attention_mask', 'labels']
    d = {k: [ex[k] for ex in batch] for k in keys}
    d['input_ids'] = torch.nn.utils.rnn.pad_sequence(d['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id)
    d['attention_mask'] = torch.nn.utils.rnn.pad_sequence(d['attention_mask'], batch_first=True, padding_value=0)
    d['labels'] = torch.stack(d['labels'])
    return d

# --- Quant + PEFT ---
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_type=torch.float16,
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
)

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['q_proj','v_proj','o_proj'],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
    modules_to_save=["score"],   # keep classifier head trainable
)

model = get_peft_model(model, lora_config)
model.config.pad_token_id = tokenizer.eos_token_id

# --- Metrics ---
def compute_metrics(p):
    logits = p.predictions
    labels = p.label_ids
    # threshold at 0 logit == 0.5 sigmoid
    preds = (logits > 0).astype(int)
    return {
        'f1_micro':    f1_score(labels, preds, average='micro',    zero_division=0),
        'f1_macro':    f1_score(labels, preds, average='macro',    zero_division=0),
        'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
    }


# --- Trainer with weighted BCE ---
class MultilabelTrainer(Trainer):
    def __init__(self, label_weights, **kwargs):
        super().__init__(**kwargs)
        self.label_weights = label_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # [batch, num_labels]

        pos_weight = self.label_weights.to(logits.device)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels.to(torch.float32),
            pos_weight=pos_weight,    # per-label positive class weighting
            reduction="mean"
        )
        return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir='multilabel_classification',
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    report_to="none",
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

trainer.train()

# # Evaluate on test
# test_metrics = trainer.evaluate(test_dataset)
# print("Test metrics:", test_metrics)

# Save PEFT adapters + tokenizer
peft_model_id = 'multilabel_gemma4b_it_lora_rus'
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
