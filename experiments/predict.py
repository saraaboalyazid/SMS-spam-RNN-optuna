import json
import os
import sys
import torch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import SMSDataset, df
from models.rnn_pad_packed import RNNPadPacked

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = os.path.dirname(__file__)
PARAMS_PATH = os.path.join(base_dir, "best_params.json")

MODEL_PATH = "../final_best_model.pth"
 


def build_vocab():
    """
    Rebuild vocab exactly like training
    """
    dataset = SMSDataset(df["text"], df["label"])
    return dataset.vocab


def load_model(vocab_size):
    """
    Recreate model architecture and load weights
    """
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

def encode_text(text, vocab):
    encoded = [vocab.get(ch, 0) for ch in text]
    tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([len(encoded)], dtype=torch.long).to(DEVICE)
    return tensor, lengths

def predict(text, model, vocab):
    encoded, lengths = encode_text(text, vocab)

    with torch.no_grad():
        logits = model(encoded, lengths)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    label = "spam" if pred_class == 1 else "ham"
    confidence = probs[0, pred_class].item()

    return label, confidence


def main():
    if len(sys.argv) < 2:
        print('Usage: python experiments/predict.py "Your SMS text here"')
        sys.exit(1)

    text = sys.argv[1]

    vocab = build_vocab()
    model = load_model(len(vocab))

    label, confidence = predict(text, model, vocab)

    print("\n===== Prediction =====")
    print(f"Text: {text}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
