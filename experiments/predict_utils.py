import json
import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import SMSDataset, df
from models.rnn_pad_packed import RNNPadPacked

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "final_best_model.pth")
PARAMS_PATH = os.path.join(BASE_DIR, "best_params.json")

dataset = SMSDataset(df["text"], df["label"])
VOCAB = dataset.vocab

with open(PARAMS_PATH, "r") as f:
    BEST_PARAMS = json.load(f)

MODEL = RNNPadPacked(
    vocab_size=len(VOCAB),
    emb_size=BEST_PARAMS["emb_size"],
    hidden_size=BEST_PARAMS["hidden_size"],
    output_size=2,
    dropout=BEST_PARAMS["dropout"],
    bidirectional=True,
).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
MODEL.load_state_dict(checkpoint["model_state_dict"])
MODEL.eval()


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def encode_text(text: str):
    encoded = [VOCAB.get(ch, 0) for ch in text]
    tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([len(encoded)], dtype=torch.long).to(DEVICE)
    return tensor, lengths


def predict_sms(text: str):
    x, lengths = encode_text(text)

    with torch.no_grad():
        logits = MODEL(x, lengths)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return {
        "label": "spam" if pred == 1 else "ham",
        "confidence": float(probs[0, pred]),
        "spam_probability": float(probs[0, 1]),
    }
