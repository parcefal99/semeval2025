# SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Detection  
**Encoder-based BERT + Decoder-based Models + Prompt Optimization + Ensembling**

We provide a utility script to generate basic distribution reports over the dataset.

To compute dataset statistics, run:

```bash
python notebooks/dataset_stat.py
```

### XLM-RoBERTa training
To run the whole baseline XLM-RoBERTa pipeline (training + per label threshold tuning + evaluation) simply run:

```bash
python notebooks/bert_ft.py
```

The trained model can be found here: https://huggingface.co/yltyadi/multilabel-emotion-detection-bert

### Fine tuning and path configuration
Edit the configuration in your training script to match your local environment in ```qlora_ft.py ``` file:

1. **Model Directory** (`MODEL_PATH`)
   - Location: `/l/users/abzal.nurgazy/models/llama32_3b`
2. **Dataset Directory** (`DATASET_PATH`)
   - Location: `/home/abzal.nurgazy/semeval2025/dataset`
3. **Training File** (`TRAIN_PATH`)
   - Filename: `esp_train.csv` (Switch to `rus_train.csv` or `deu_train.csv` as needed)
4. **Validation File** (`DEV_PATH`)
   - Filename: `esp_dev.csv`

### Testing fine-tuned models 

To test the fine-tuned models, run:
```bash
python notebooks/test_all_qlora.py
```

Fine-tuned qlora adapters can be found with link: 
1. https://huggingface.co/parcefal99/semeval25-task11-adapters/tree/main
