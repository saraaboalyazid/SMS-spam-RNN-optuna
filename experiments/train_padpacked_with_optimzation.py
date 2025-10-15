# Step 1: Create a collate_fn for your DataLoader

import os
import sys

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import (PackedSequence, pack_padded_sequence,
                                pad_sequence)
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataset import SMSDataset, df
from framework.training.trainer import train_network
from models.rnn_pad_packed import RNNPadPacked


def collate_fn(batch):
    """
    Collate function for padding variable-length sequences and returning a PackedSequence.

    Args:
        batch: list of tuples (sequence_tensor, label)
               - sequence_tensor: torch.Tensor of shape (seq_len,)
               - label: torch.Tensor scalar or int

    Returns:
        padded_seqs: torch.Tensor of shape (B, T)
        labels: torch.Tensor of shape (B,)
        lengths: torch.Tensor of original sequence lengths (B,)
    """

    sequences, labels = zip(*batch)  # unzip the batch list

    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

    padded_seqs = pad_sequence(
        sequences, batch_first=True, padding_value=0
    )  # shape → (B, T) where T = max sequence length in batch
    x_packed = pack_padded_sequence(
        padded_seqs, lengths, batch_first=True, enforce_sorted=False
    )

    labels = torch.tensor(labels, dtype=torch.long)

    return x_packed, labels, lengths


# Step 2: Create DataLoaders for training/validation/testing
dataset = SMSDataset(df["text"], df["label"])

# Define split ratios
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2
# Compute split sizes
total_size = len(dataset)
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size  # ensures all data used

# Randomly split the dataset
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

# DataLoaders
B = 32
train_loader = DataLoader(
    train_dataset, batch_size=B, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(val_dataset, batch_size=B, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(
    test_dataset, batch_size=B, shuffle=False, collate_fn=collate_fn
)

# -------------------------------
# DEBUG: Check one batch of data
# x, y, lengths = next(iter(train_loader))  # or x, y if your collate_fn returns only padded_seqs and labels
# print("Input batch shape:", x.shape)
# print("Labels batch shape:", y.shape)
# print("Sequence lengths:", lengths)
# print("Labels in this batch:", y)

# # Optional: check for invalid labels
# assert y.min() >= 0 and y.max() < 2, "Labels are out of bounds!"


# Step 3: Define the training loop
# 1. Device setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 2. Model, loss function, optimizer
loss_func = nn.CrossEntropyLoss()
vocab_size = len(dataset.vocab)
model = RNNPadPacked(
    vocab_size=vocab_size,
    emb_size=32,
    hidden_size=64,
    output_size=2,
    bidirectional=True,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.2, patience=10
)

third_model = train_network(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    loss_func=loss_func,
    device=device,
    lr_schedule=scheduler,
    epochs=10,
    score_funcs={"Accuracy": accuracy_score},
    checkpoint_file="rnn_pad_packed_bidrec_with_optimization.pth",
)
