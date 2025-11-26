import os
import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import roc_auc_score, roc_curve, auc
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


DATASET_PATH = "/home/abzal.nurgazy/semeval2025/dataset"
ADAPTER_DIR = "/home/abzal.nurgazy/semeval2025/notebooks"
OUTPUT_IMAGE_NAME = "deu_gemma4b.png" 

plt.switch_backend('Agg')

BASE_PHI = "/l/users/abzal.nurgazy/models/phy_3b_mini"
BASE_LLAMA = "/l/users/abzal.nurgazy/models/llama32_3b"


models_to_eval = {
    # # --- PHI-3 ---
    # "Phi-3 (All)": {
    #     "base_path": BASE_PHI, 
    #     "adapter_path": os.path.join(ADAPTER_DIR, "multilabel_phy_3b_lora_all"),
    #     "test_file": "all_test.csv"
    # },
    # "Phi-3 (Esp)": {
    #     "base_path": BASE_PHI, 
    #     "adapter_path": os.path.join(ADAPTER_DIR, "multilabel_phy_3b_esp"),
    #     "test_file": "esp_test.csv"
    # },
    # "Phi-3 (Deu)": {
    #     "base_path": BASE_PHI, 
    #     "adapter_path": os.path.join(ADAPTER_DIR, "multilabel_phy_3b_deu"),
    #     "test_file": "deu_test.csv"
    # },
    # "Phi-3 (Rus)": {
    #     "base_path": BASE_PHI, 
    #     "adapter_path": os.path.join(ADAPTER_DIR, "multilabel_phy_3b_rus"),
    #     "test_file": "rus_test.csv"
    # },
    
    # # --- LLAMA-3.2 ---
    # "Llama-3.2 (All)": {
    #     "base_path": BASE_LLAMA, 
    #     "adapter_path": os.path.join(ADAPTER_DIR, "multilabel_llama32_3b_lora_all"),
    #     "test_file": "all_test.csv"
    # },
    # "Llama-3.2 (Esp)": {
    #     "base_path": BASE_LLAMA, 
    #     "adapter_path": os.path.join(ADAPTER_DIR, "multilabel_llama32_3b_lora_esp"),
    #     "test_file": "esp_test.csv"
    # },
    # "Llama-3.2 (Deu)": {
    #     "base_path": BASE_LLAMA, 
    #     "adapter_path": os.path.join(ADAPTER_DIR, "multilabel_llama32_3b_lora_deu"),
    #     "test_file": "deu_test.csv"
    # },
    # "Llama-3.2 (Rus)": {
    #     "base_path": BASE_LLAMA, 
    #     "adapter_path": os.path.join(ADAPTER_DIR, "multilabel_llama32_3b_lora_rus"),
    #     "test_file": "rus_test.csv"
    # },
    "Gemma-4 (Deu)": {
        "base_path": "/l/users/abzal.nurgazy/models/gemma4b", 
        "adapter_path": os.path.join(ADAPTER_DIR, "multilabel_gemma4b_test_deu"),
        "test_file": "deu_test.csv"
    },
}


class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, csv_path, tokenizer):
        file_full_path = os.path.join(dataset_path, csv_path)
        if not os.path.exists(file_full_path):
            raise FileNotFoundError(f"CSV not found: {file_full_path}")
        self.data = pd.read_csv(file_full_path, encoding='utf-8')
        self.tokenizer = tokenizer
        self.label_cols = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text']
        labels = row[self.label_cols].values.astype(np.float32)
        enc = self.tokenizer(text, truncation=True, max_length=512, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0), 'attention_mask': enc['attention_mask'].squeeze(0), 'labels': torch.tensor(labels, dtype=torch.float32)}

def get_dataloader(model_path, csv_filename, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None: tokenizer.pad_token = tokenizer.eos_token
        else: tokenizer.add_special_tokens({"pad_token": "<pad>"})
    
    dataset = EmotionDataset(DATASET_PATH, csv_filename, tokenizer)

    def collate_fn(batch):
        input_ids_list = [x['input_ids'] for x in batch]
        labels = torch.stack([x['labels'] for x in batch])
        batch_enc = tokenizer.pad({'input_ids': input_ids_list}, padding=True, return_tensors='pt')
        return {'input_ids': batch_enc['input_ids'], 'attention_mask': batch_enc['attention_mask'], 'labels': labels}

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=2), dataset.label_cols, tokenizer


def run_inference(name, config):
    csv_file = config['test_file']
    print(f"  > Processing: {name} (on {csv_file})")
    
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_type=torch.bfloat16)
    test_loader, label_names, tokenizer = get_dataloader(config['base_path'], csv_file)

    model = AutoModelForSequenceClassification.from_pretrained(
        config['base_path'], num_labels=len(label_names), quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True, ignore_mismatched_sizes=True, attn_implementation="eager", 
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.problem_type = "multi_label_classification"

    try:
        model = PeftModel.from_pretrained(model, config['adapter_path'])
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"    !!!! SKIPPING {name} due to dimension mismatch !!!!")
            return None, None
        raise e
    except Exception as e:
        print(f"    !!!! SKIPPING {name}: {e}")
        return None, None

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
            outputs = model(**inputs)
            all_probs.append(torch.sigmoid(outputs.logits.float()).cpu().numpy())
            all_labels.append(batch['labels'].cpu().numpy())

    y_probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    
    del model, tokenizer, test_loader
    torch.cuda.empty_cache()
    gc.collect()
    return y_true, y_probs


if __name__ == "__main__":
    
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Cross-Lingual Model Evaluation (All Models)', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)

    
    model_names_list = list(models_to_eval.keys())
    cmap = cm.get_cmap('tab10') # or 'Set1'
    model_colors = {name: cmap(i) for i, name in enumerate(model_names_list)}

    print(f"\n=== STARTING EVALUATION ===")
    print(f"Monitor the progress by opening: {OUTPUT_IMAGE_NAME}\n")

    for model_name, conf in models_to_eval.items():
        if not os.path.exists(conf['adapter_path']):
            print(f"Skipping {model_name} (Adapter not found)")
            continue

        try:

            y_true, y_probs = run_inference(model_name, conf)
            
            if y_true is None: continue

            scores = y_probs.ravel()
            labels = y_true.ravel()
            mask = np.isfinite(scores)
            
            if mask.sum() > 0:
                fpr, tpr, _ = roc_curve(labels[mask], scores[mask])
                roc_auc = auc(fpr, tpr)
                
                color = model_colors.get(model_name, 'black')
                
                linestyle = '--' if "(All)" in model_name else '-'
                
                ax.plot(fpr, tpr, color=color, lw=2, linestyle=linestyle, 
                        label=f'{model_name} (AUC={roc_auc:.2f})')
                
                ax.legend(loc='lower right', fontsize=10)
                plt.savefig(OUTPUT_IMAGE_NAME, dpi=300, bbox_inches='tight')
                print(f"  >>> Updated plot: {OUTPUT_IMAGE_NAME}")

        except Exception as e:
            print(f"CRITICAL ERROR on {model_name}: {e}")
            torch.cuda.empty_cache()

    print(f"\nFinished! Final plot saved as {OUTPUT_IMAGE_NAME}")