
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import json
from sklearn.metrics import accuracy_score
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataset import SMSDataset, df
from models.rnn_pad_packed import RNNPadPacked
from framework.training.trainer import train_network
from train_with_optuna import collate_fn  , train_dataset, val_dataset, test_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SMSDataset(df["text"], df["label"])

base_dir = os.path.dirname(__file__)
params_path = os.path.join(base_dir, "best_params.json")

with open(params_path, "r") as f:
    best_params = json.load(f)

full_train_dataset = ConcatDataset([train_dataset, val_dataset])
full_train_loader = DataLoader(full_train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

vocab_size = len(dataset.vocab)
best_model = RNNPadPacked(
    vocab_size=vocab_size,
    emb_size=best_params["emb_size"],
    hidden_size=best_params["hidden_size"],
    output_size=2,
    dropout=best_params["dropout"],
    bidirectional=True
).to(device)

optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params["learning_rate"])


print("\n Retraining final model on full dataset...")
final_results = train_network(
    model=best_model,
    loss_func=nn.CrossEntropyLoss(),
    train_loader=full_train_loader,
    val_loader=None,
    test_loader=test_loader,
    optimizer=optimizer,
    device=device,
    epochs=30,
    score_funcs={"Accuracy": accuracy_score},
    checkpoint_file="final_best_model.pth"
)

print("\nFinal training complete. Model saved as final_best_model.pth")
