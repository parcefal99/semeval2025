import os, json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from torch.utils.data import DataLoader

EMOTIONS = ["anger","disgust","fear","joy","sadness","surprise"]
MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 256
SEED = 42
LANGS = ["eng", "rus", "esp", "deu", "all"]
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "dataset/Part_A"
OUT_ROOT = BASE_DIR / "enc_emotions"
os.makedirs(OUT_ROOT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def df_to_hf(df):
    return Dataset.from_pandas(df[["text"] + EMOTIONS], preserve_index=False)

def tokenize(batch):
    enc = tok(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    labels = np.stack([np.asarray(batch[e], dtype=np.float32) for e in EMOTIONS], axis=1)
    enc["labels"] = labels
    return enc

def compute_metrics(p):
    logits = p.predictions
    if logits.ndim == 3:              # sometimes (N,1,C)
        logits = logits.squeeze(1)
    probs = 1/(1+np.exp(-logits))
    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    preds = (probs >= 0.5).astype(int)  # temp; thresholds later
    macro = f1_score(p.label_ids, preds, average="macro", zero_division=0)
    return {"macro_f1": macro}

def load_dataset(lang):
    train_csv = DATA_DIR / f"{lang}_train.csv"
    dev_csv = DATA_DIR / f"{lang}_dev.csv"
    test_csv = DATA_DIR / f"{lang}_test.csv"

    train_df = pd.read_csv(train_csv)
    dev_df   = pd.read_csv(dev_csv)

    hf_train = df_to_hf(train_df).map(tokenize, batched=True, remove_columns=["text"]+EMOTIONS)
    hf_dev = df_to_hf(dev_df).map(tokenize, batched=True, remove_columns=["text"]+EMOTIONS)

    return train_df, dev_df, hf_train, hf_dev


def main():
    set_seed(SEED)

    summaries = []

    for lang in LANGS:
        out_dir = OUT_ROOT / lang
        os.makedirs(out_dir, exist_ok=True)
        train_df, dev_df, hf_train, hf_dev = load_dataset(lang)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(EMOTIONS),
            problem_type="multi_label_classification"  # BCEWithLogitsLoss inside
        )
        collator = DataCollatorWithPadding(tokenizer=tok)

        # JUST CHECKING BEFORE RUNNING
        # model.eval()
        # batch = next(iter(DataLoader(hf_dev, batch_size=4, collate_fn=collator)))
        # with torch.no_grad():
        #     out = model(**{k: v.to(model.device) for k, v in batch.items()})
        # print(f"[{lang}] dev forward loss:", out.loss)

        # TRAINING PART
        args = TrainingArguments(
            output_dir=str(out_dir),
            learning_rate=1e-5,
            max_grad_norm=1.0,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=4,
            fp16=False, bf16=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            logging_steps=50,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=hf_train,
            eval_dataset=hf_dev,
            processing_class=tok,
            data_collator=collator,
            compute_metrics=compute_metrics
        )

        trainer.train()

        # save best model + tokenizer + logs
        best_dir = out_dir / "best_model"
        trainer.save_model(best_dir)
        tok.save_pretrained(best_dir)

        eval_logs = [r for r in trainer.state.log_history if "eval_loss" in r]
        with open(out_dir / "epoch_logs.jsonl", "w", encoding="utf-8") as f:
            for r in eval_logs:
                f.write(json.dumps(r) + "\n")
        for r in eval_logs:
            ep = int(r["epoch"])
            print(f"Epoch {ep}: val_loss={r['eval_loss']:.6f}, macro_f1={r.get('eval_macro_f1', 0):.6f}")

        # threshold tuning on dev
        dev_logits = trainer.predict(hf_dev).predictions
        if dev_logits.ndim == 3:
            dev_logits = dev_logits.squeeze(1)
        dev_probs = 1/(1+np.exp(-dev_logits))
        dev_probs = np.nan_to_num(dev_probs, nan=0.0, posinf=1.0, neginf=0.0)
        dev_true = np.stack([dev_df[e].values for e in EMOTIONS], axis=1).astype(int)

        best_t = []
        for j in range(len(EMOTIONS)):
            ts = np.linspace(0.05, 0.95, 37)
            f1s = [f1_score(dev_true[:, j], (dev_probs[:, j] >= t).astype(int), zero_division=0) for t in ts]
            best_t.append(float(ts[int(np.argmax(f1s))]))

        macro_tuned = f1_score(dev_true, (dev_probs >= np.array(best_t)).astype(int), average="macro", zero_division=0)
        print(f"[{lang}] Best thresholds:", dict(zip(EMOTIONS, best_t)), "dev macro-F1:", macro_tuned)

        # save thresholds
        with open(out_dir / "thresholds.json","w",encoding="utf-8") as f:
            json.dump(dict(zip(EMOTIONS, best_t)), f, indent=2)

if __name__ == "__main__":
    main()
