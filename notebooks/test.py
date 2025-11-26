import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


TEST_PATH = "esp_test.csv"
DATASET_PATH = "/home/abzal.nurgazy/semeval2025/dataset"
ADAPTER_DIR = "/home/abzal.nurgazy/semeval2025/notebooks"
OUTPUT_IMAGE_NAME = "comparison_roc_curves.png"

models_to_eval = {
    "Phi-3 LoRA": {
        "base_path": "/l/users/abzal.nurgazy/models/phy_3b_mini",
        "adapter_path": os.path.join(ADAPTER_DIR, "multilabel_phy_3b_esp")
    },
    "Llama-3.2 LoRA": {
        # CHECK THIS PATH:
        "base_path": "/l/users/abzal.nurgazy/models/llama32_3b", 
        "adapter_path": os.path.join(ADAPTER_DIR, "multilabel_llama32_3b_lora_esp")
    },
    # "Gemma-2 LoRA": {
    #     # CHECK THIS PATH:
    #     "base_path": "/l/users/abzal.nurgazy/models/gemma-2-2b",   
    #     "adapter_path": os.path.join(ADAPTER_DIR, "multilabel_gemma4b_lora_all")
    # }
}


class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, csv_path, tokenizer, max_length=512):
        file_full_path = os.path.join(dataset_path, csv_path)
        if not os.path.exists(file_full_path):
            raise FileNotFoundError(f"CSV file not found at: {file_full_path}. Please check DATASET_PATH.")
        self.data = pd.read_csv(file_full_path, encoding='utf-8')
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_cols = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text']
        labels = row[self.label_cols].values.astype(np.float32)

        enc = self.tokenizer(
            text,
            truncation=True, 
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }

def get_dataloader(model_path, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    
    tokenizer.padding_side = "left" 
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = EmotionDataset(DATASET_PATH, TEST_PATH, tokenizer)

    def collate_fn(batch):
        input_ids_list = [x['input_ids'] for x in batch]
        labels = torch.stack([x['labels'] for x in batch])
        
        batch_enc = tokenizer.pad(
            {'input_ids': input_ids_list},
            padding=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': batch_enc['input_ids'],
            'attention_mask': batch_enc['attention_mask'],
            'labels': labels
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=2), dataset.label_cols, tokenizer


def run_inference(name, config):
    print(f"\n--- Running Inference: {name} ---")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_type=torch.bfloat16  
    )

    test_loader, label_names, tokenizer = get_dataloader(config['base_path'])

    print(f"Loading Base Model: {config['base_path']}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config['base_path'],
        num_labels=len(label_names),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        attn_implementation="eager", 
    )


    model.config.pad_token_id = tokenizer.pad_token_id

    model.config.problem_type = "multi_label_classification"

    print(f"Loading Adapter: {config['adapter_path']}")
    model = PeftModel.from_pretrained(model, config['adapter_path'])
    model.eval()

    all_probs, all_labels = [], []

    print("Starting batch prediction...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}

            outputs = model(**inputs)
            
            logits = outputs.logits.float()
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_labels.append(batch['labels'].cpu().numpy())
            if (i + 1) % 10 == 0:
                print(f"Processed batch {i+1}")

    y_probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    
    nan_count = np.isnan(y_probs).sum()
    if nan_count > 0:
        print(f"WARNING: Found {nan_count} NaNs in predictions. Check adapter weights.")
    else:
        print("Success: No NaNs detected.")

    y_probs = np.nan_to_num(y_probs, nan=0.5, posinf=1.0, neginf=0.0)

    del model
    torch.cuda.empty_cache()
    print(f"Finished inference for {name}. VRAM cleared.")

    return y_true, y_probs

if __name__ == "__main__":
    model_results = {}

    for model_name, conf in models_to_eval.items():
        try:
            true_labels, pred_probs = run_inference(model_name, conf)
            model_results[model_name] = (true_labels, pred_probs)
        except Exception as e:
            print(f"Error running {model_name}: {e}")

    print("\nGenerating Comparison ROC Plot...")
    plt.figure(figsize=(10, 8))

    colors = ['blue', 'green', 'red']
    plotted_any = False

    for i, (model_name, (y_true, y_probs)) in enumerate(model_results.items()):
        scores = y_probs.ravel()
        labels = y_true.ravel()

        mask = np.isfinite(scores)
        if mask.sum() == 0:
            print(f"Skipping {model_name}: no finite scores.")
            continue

        fpr, tpr, _ = roc_curve(labels[mask], scores[mask])
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr, tpr,
            color=colors[i % len(colors)],
            lw=2,
            label=f'{model_name} (Micro AUC = {roc_auc:.3f})'
        )
        plotted_any = True

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Comparison of Micro-Average ROC Curves (QLoRA Models)', fontsize=14)

    if plotted_any:
        plt.legend(loc="lower right", fontsize=11)
    else:
        print("No curves to plot.")

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(OUTPUT_IMAGE_NAME, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully as: {OUTPUT_IMAGE_NAME}")