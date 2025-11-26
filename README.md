# SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Detection  
**Encoder-based BERT + Decoder-based Models + Prompt Optimization + Ensembling**

We provide a utility script to generate basic distribution reports over the dataset.

To compute dataset statistics, run:

```bash
python scripts/dataset_stat.py
```

Paths and Configuration
Edit the configuration in your training script to match your local environment:

MODEL_PATH   = "/l/users/abzal.nurgazy/models/llama32_3b"
DATASET_PATH = "/home/abzal.nurgazy/semeval2025/dataset"
TRAIN_PATH = "esp_train.csv"
DEV_PATH   = "esp_dev.csv"

