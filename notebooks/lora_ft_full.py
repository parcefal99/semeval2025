import os, functools, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix



LANGUAGE = sys.argv[1] if len(sys.argv) > 1 else "all"
MODEL_PATH = "/l/users/zhalgas.bekeyev/models/llama-3.2-3b"
DATASET_PATH = "/l/users/zhalgas.bekeyev/projects/NLP701/dataset"

LANG_CONFIGS = {
    "deu": {"train": "deu_train.csv", "dev": "deu_dev.csv", "test": "deu_test.csv"},
    "esp": {"train": "esp_train.csv", "dev": "esp_dev.csv", "test": "esp_test.csv"},
    "rus": {"train": "rus_train.csv", "dev": "rus_dev.csv", "test": "rus_test.csv"},
}

print(f"Training: {LANGUAGE.upper()}")
print(f"Model: LLama-3.2-3b with LoRA (BF16)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


import os

def print_trainable_parameters(model, language, save_dir="/l/users/zhalgas.bekeyev/projects/NLP701/results/llama"):
    trainable_params = 0
    all_params = 0

    # Count params
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    percent = 100 * trainable_params / all_params
    summary = (
        f"Language: {language.upper()}\n"
        f"Trainable params: {trainable_params:,}\n"
        f"All params: {all_params:,}\n"
        f"Trainable %: {percent:.2f}%\n"
    )

    print(summary)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"trainable_params_{language}.txt")

    with open(save_path, "w") as f:
        f.write(summary)

    print(f"Saved trainable parameter info to: {save_path}")
    
    
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, csv_path, tokenizer, max_length=512):
        self.data = pd.read_csv(os.path.join(dataset_path, csv_path), encoding='utf-8')
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_cols = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
        self.num_labels = len(self.label_cols)
        
        missing_cols = [col for col in self.label_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in {csv_path}: {missing_cols}")
        
        self.label_weights = 1 - self.data[self.label_cols].sum(axis=0) / self.data[self.label_cols].sum().sum()
        print(f"  Loaded {len(self.data)} samples")

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


if LANGUAGE == "all":
    train_dfs = []
    dev_dfs = []
    test_dfs = []
    for lang, paths in LANG_CONFIGS.items():
        train_dfs.append(pd.read_csv(os.path.join(DATASET_PATH, paths["train"])))
        dev_dfs.append(pd.read_csv(os.path.join(DATASET_PATH, paths["dev"])))
        if paths["test"]:
            test_dfs.append(pd.read_csv(os.path.join(DATASET_PATH, paths["test"])))
    
    combined_train = pd.concat(train_dfs, ignore_index=True)
    combined_dev = pd.concat(dev_dfs, ignore_index=True)
    combined_test = pd.concat(test_dfs, ignore_index=True) if test_dfs else combined_dev
    
    os.makedirs(os.path.join(DATASET_PATH, "combined"), exist_ok=True)
    combined_train.to_csv(os.path.join(DATASET_PATH, "combined/all_train.csv"), index=False)
    combined_dev.to_csv(os.path.join(DATASET_PATH, "combined/all_dev.csv"), index=False)
    combined_test.to_csv(os.path.join(DATASET_PATH, "combined/all_test.csv"), index=False)
    
    train_path = "combined/all_train.csv"
    dev_path = "combined/all_dev.csv"
    test_path = "combined/all_test.csv"
else:
    train_path = LANG_CONFIGS[LANGUAGE]["train"]
    dev_path = LANG_CONFIGS[LANGUAGE]["dev"]
    test_path = LANG_CONFIGS[LANGUAGE]["test"]

print(f"\nLoading datasets for {LANGUAGE.upper()}...")
train_dataset = EmotionDataset(DATASET_PATH, train_path, tokenizer)
dev_dataset = EmotionDataset(DATASET_PATH, dev_path, tokenizer)

if test_path:
    test_dataset = EmotionDataset(DATASET_PATH, test_path, tokenizer)
else:
    print("No test set available (using dev for final evaluation)")
    test_dataset = dev_dataset


def collate_fn(batch, tokenizer):
    keys = ['input_ids', 'attention_mask', 'labels']
    d = {k: [ex[k] for ex in batch] for k in keys}
    d['input_ids'] = torch.nn.utils.rnn.pad_sequence(d['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id)
    d['attention_mask'] = torch.nn.utils.rnn.pad_sequence(d['attention_mask'], batch_first=True, padding_value=0)
    d['labels'] = torch.stack(d['labels'])
    return d


config = AutoConfig.from_pretrained(
    MODEL_PATH,
    num_labels=train_dataset.num_labels,
    problem_type="multi_label_classification",
    trust_remote_code=True,
    local_files_only=True
)

print("\nLoading model in BF16...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
)

lora_config = LoraConfig(
    r=64, 
    lora_alpha=128,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], 
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    modules_to_save=["score"],
)

model = get_peft_model(model, lora_config)
model.config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

print("\nModel Configuration:")
print_trainable_parameters(model, language=LANGUAGE)



def plot_and_save_roc(labels, probs, label_names, language, split, save_dir="/l/users/zhalgas.bekeyev/projects/NLP701/results/llama"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8,6))
    for i, label in enumerate(label_names):
        fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],'k--',lw=1)
    plt.xlim([0.0,1.0]); plt.ylim([0.0,1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - {language.upper()} ({split})")
    plt.legend(loc="lower right", fontsize="small")
    file_path = os.path.join(save_dir, f"roc_auc_{language}_{split}.png")
    plt.tight_layout(); plt.savefig(file_path, dpi=200); plt.close()
    print(f"ROC curves saved to {file_path}")

def plot_and_save_confusion_matrices(labels, preds, label_names, language, split, save_dir="/l/users/zhalgas.bekeyev/projects/NLP701/results/llama"):
    os.makedirs(save_dir, exist_ok=True)
    matrices = multilabel_confusion_matrix(labels, preds)
    num_labels = len(label_names)
    cols = 3
    rows = int(np.ceil(num_labels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.flatten()

    for i, (ax, label) in enumerate(zip(axes, label_names)):
        cm = matrices[i]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(label)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Confusion Matrices - {language.upper()} ({split})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    file_path = os.path.join(save_dir, f"confusion_{language}_{split}.png")
    plt.savefig(file_path, dpi=200)
    plt.close()
    print(f"Confusion matrices saved to {file_path}")



def compute_metrics(p):
    logits = p.predictions
    labels = p.label_ids
    probs = 1 / (1 + np.exp(-logits)) 
    preds = (probs > 0.5).astype(int)
    label_names = ["anger","disgust","fear","joy","sadness","surprise"]

    # --- F1 scores ---
    metrics = {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }

    # --- ROC AUC ---
    try:
        roc_auc_macro = roc_auc_score(labels, probs, average="macro")
        metrics["roc_auc_macro"] = roc_auc_macro
    except ValueError:
        metrics["roc_auc_macro"] = float("nan")

    label_names = ["anger","disgust","fear","joy","sadness","surprise"]
    current_split = "eval" 
    plot_and_save_roc(labels, probs, label_names, language=LANGUAGE, split=current_split)
    plot_and_save_confusion_matrices(labels, preds, label_names, language=LANGUAGE, split=current_split)

    return metrics




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
            logits,
            labels.to(torch.float32),
            pos_weight=pos_weight,
            reduction="mean"
        )
        return (loss, outputs) if return_outputs else loss


output_dir = f'/l/users/zhalgas.bekeyev/projects/NLP701/results/llama/llama_lora_{LANGUAGE}'
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    greater_is_better=True,
    report_to="none",
    bf16=True, 
    gradient_accumulation_steps=2,
    logging_steps=50,
    save_total_limit=2,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
)

trainer = MultilabelTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    data_collator=functools.partial(collate_fn, tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    label_weights=torch.tensor(train_dataset.label_weights.values, dtype=torch.float32),
)

print("Starting training...")

trainer.train()

print("Evaluating on test/dev set...")
test_metrics = trainer.evaluate(test_dataset)
print("\nFinal Test Metrics:")
for key, value in test_metrics.items():
    key_clean = key.replace('eval_', '')
    print(f"  {key_clean}: {value:.4f}")

peft_model_id = f'/l/users/zhalgas.bekeyev/projects/NLP701/models/llama_lora_{LANGUAGE}'
os.makedirs(peft_model_id, exist_ok=True)
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
print(f"\nModel saved to {peft_model_id}")

os.makedirs('/l/users/zhalgas.bekeyev/projects/NLP701/results/llama', exist_ok=True)
results_df = pd.DataFrame([{
    'language': LANGUAGE,
    'model': 'LLama-3.2-3b-LoRA',
    'test_split': 'test' if test_path else 'dev',
    **{k.replace('eval_', ''): v for k, v in test_metrics.items()}
}])

results_dir = "/l/users/zhalgas.bekeyev/projects/NLP701/results/llama"
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, "all_results.csv")


try:
    if os.path.exists(results_file):
        existing = pd.read_csv(results_file)
        results_df = pd.concat([existing, results_df], ignore_index=True)
    else:
        print(f"⚠️  all_results.csv not found — creating a new one.")
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")
except Exception as e:
    print(f"Could not save results: {e}")
