import os
import sys

import optuna
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, random_split

# Add project root to sys.path
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


# --- Dataset and Loaders ---
dataset = SMSDataset(df["text"], df["label"])

train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

B = 32
train_loader = DataLoader(
    train_dataset, batch_size=B, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(val_dataset, batch_size=B, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(
    test_dataset, batch_size=B, shuffle=False, collate_fn=collate_fn
)


# --- Optuna objective function ---
def objective(trial):
    emb_size = trial.suggest_int("emb_size", 16, 128)
    hidden_size = trial.suggest_int("hidden_size", 32, 256)
    lr = trial.suggest_loguniform("learning_rate", 1e-6, 1e2)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = len(dataset.vocab)
    model = RNNPadPacked(
        vocab_size=vocab_size,
        emb_size=emb_size,
        hidden_size=hidden_size,
        output_size=2,
        bidirectional=True,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=5
    )
    loss_func = nn.CrossEntropyLoss()
    score_funcs = {"acc": lambda y_pred, y_true: accuracy_score(y_true, y_pred)}
    history = train_network(
        model,
        loss_func,
        train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        score_funcs=score_funcs,
        epochs=10,
        device=device,
        optimizer=optimizer,
        lr_schedule=scheduler,
    )
    print("History columns:", history.columns)
    print(history.tail())

    val_acc = history["test acc"].iloc[-1]
    return val_acc


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Accuracy: {trial.value}")
    print("  Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
