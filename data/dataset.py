import pandas as pd
import torch
from torch.utils.data import Dataset
import os

base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "spam.csv")
print("CSV path:", csv_path)

# Load dataset safely
df_raw = pd.read_csv(csv_path, encoding="latin-1")
# print("Columns in CSV:", df_raw.columns)
# print(df_raw.head())

# Keep only the relevant columns
df = df_raw[["v1", "v2"]].copy()
df.columns = ["label", "text"]

# Map labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Drop rows where label mapping failed
df = df.dropna(subset=['label']).reset_index(drop=True)

# Convert labels to integer type
df['label'] = df['label'].astype(int)

# print("Number of samples:", len(df))
# print("Labels:", df['label'].unique())

# Dataset class
class SMSDataset(Dataset):
    def __init__(self, texts, labels, vocab=None):
        if vocab is None:
            all_text = "".join(texts.tolist())
            vocab = {"<PAD>": 0}
            vocab.update({ch: i + 1 for i, ch in enumerate(sorted(set(all_text)))})
        self.vocab = vocab
        self.data = [
            torch.tensor([self.vocab.get(ch, 0) for ch in text], dtype=torch.long)
            for text in texts
        ]
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = SMSDataset(df['text'], df['label'])
# print("Dataset length:", len(dataset))
# print("First sample:", dataset[0])
# print("Vocabulary size:", len(dataset.vocab))