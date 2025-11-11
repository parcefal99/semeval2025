import pandas as pd
from pathlib import Path

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "dataset"

# Process eng_dev.csv
df_dev = pd.read_csv(DATA_DIR / 'eng_dev.csv')
# Insert 'disgust' column at index 3 (after 'anger' at index 2) with value 0
df_dev.insert(loc=3, column='disgust', value=0)
df_dev.to_csv(DATA_DIR / 'eng_dev.csv', index=False)

# Process eng_train.csv
df_train = pd.read_csv(DATA_DIR / 'eng_train.csv')
# Insert 'disgust' column at index 3 (after 'anger' at index 2) with value 0
df_train.insert(loc=3, column='disgust', value=0)
df_train.to_csv(DATA_DIR / 'eng_train.csv', index=False)