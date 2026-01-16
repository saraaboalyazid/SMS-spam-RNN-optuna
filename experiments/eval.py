import json
import os
import sys
import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import SMSDataset, df
from models.rnn_pad_packed import RNNPadPacked
from train_with_optuna import collate_fn, test_dataset
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# paths
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "final_best_model.pth")
PARAMS_PATH = os.path.join(BASE_DIR, "best_params.json")


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def build_vocab():
    dataset = SMSDataset(df["text"], df["label"])
    return dataset.vocab


def load_model(vocab_size):
    with open(PARAMS_PATH, "r") as f:
        best_params = json.load(f)

    model = RNNPadPacked(
        vocab_size=vocab_size,
        emb_size=best_params["emb_size"],
        hidden_size=best_params["hidden_size"],
        output_size=2,
        dropout=best_params["dropout"],
        bidirectional=True,
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(model, dataloader):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for texts, labels, lengths in dataloader:
            texts = texts.to(DEVICE)
            lengths = lengths.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(texts, lengths)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred, average="weighted"),
    "recall": recall_score(y_true, y_pred, average="weighted"),
    "f1_score": f1_score(y_true, y_pred, average="weighted"),
    "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    "classification_report": classification_report(y_true, y_pred),
}



    return metrics


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("\nRunning evaluation on test set...\n")

    vocab = build_vocab()
    model = load_model(len(vocab))

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
    )

    metrics = evaluate(model, test_loader)

    print("===== Evaluation Results =====")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1_score']:.4f}")

    print("\nConfusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))

    print("\nClassification Report:")
    print(metrics["classification_report"])

    with open("evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nMetrics saved to evaluation_metrics.json")


if __name__ == "__main__":
    main()
