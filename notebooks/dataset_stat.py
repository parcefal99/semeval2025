import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATASET_PATH = "/home/abzal.nurgazy/semeval2025/dataset"
OUTPUT_IMAGE = "split_distribution_by_language.png"
LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

LANG_CONFIG = {
    "Russian": {
        "Train": "rus_train.csv",
        "Dev":   "rus_dev.csv",
        "Test":  "rus_test.csv"
    },
    "Spanish": {
        "Train": "esp_train.csv",
        "Dev":   "esp_dev.csv",
        "Test":  "esp_test.csv"
    },
    "German": {
        "Train": "deu_train.csv",
        "Dev":   "deu_dev.csv",
        "Test":  "deu_test.csv"
    }
}

def analyze_splits_per_language():
    
    all_data = []

    print(f"{'Lang':<8} | {'Split':<6} | {'Emotion':<10} | {'Count':<5} | {'%':<5}")
    print("-" * 50)

    for lang, splits in LANG_CONFIG.items():
        for split_name, filename in splits.items():
            file_path = os.path.join(DATASET_PATH, filename)
            
            if not os.path.exists(file_path):
                print(f"  [MISSING] {lang} {split_name}: {filename}")
                continue

            try:
                df = pd.read_csv(file_path)
                total = len(df)
                
                counts = df[LABELS].sum()
                
                for label, count in counts.items():
                    pct = (count / total) * 100
                    
                    print(f"{lang:<8} | {split_name:<6} | {label:<10} | {count:<5} | {pct:.1f}%")
                    
                    all_data.append({
                        "Language": lang,
                        "Split": split_name,
                        "Emotion": label.capitalize(),
                        "Percentage": pct,
                        "Count": count
                    })
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    if all_data:
        df_plot = pd.DataFrame(all_data)
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
        sns.set_theme(style="whitegrid")

        languages = ["Russian", "Spanish", "German"]
        
        for i, lang in enumerate(languages):
            ax = axes[i]
            lang_data = df_plot[df_plot["Language"] == lang]
            
            if lang_data.empty:
                ax.set_title(f"{lang} (No Data)")
                continue

            sns.barplot(
                data=lang_data,
                x="Emotion",
                y="Percentage",
                hue="Split",
                palette="viridis", 
                ax=ax
            )
            
            ax.set_title(f"{lang} Distribution", fontsize=14, fontweight='bold')
            ax.set_xlabel("")
            if i == 0:
                ax.set_ylabel("Percentage of Split (%)")
            else:
                ax.set_ylabel("") 
            
            for container in ax.containers:
                ax.bar_label(container, fmt='%.0f', padding=2, fontsize=8)

        plt.tight_layout()
        plt.savefig(OUTPUT_IMAGE, dpi=300)
        
    else:
        print("No valid data found to plot.")

if __name__ == "__main__":
    analyze_splits_per_language()
