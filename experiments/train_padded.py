# Step 1: Create a collate_fn for your DataLoader
# collate_fn pseudocode
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import accuracy_score
import sys , os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from data.dataset import SMSDataset, df
from framework.training.trainer import train_network
from models.rnn_padded import RNNPadded
def collate_fn(batch):
    """
    Collate function for padding variable-length sequences in a batch.
    
    Args:
        batch: list of tuples (sequence_tensor, label)
               - sequence_tensor: torch.Tensor of shape (seq_len,)
               - label: torch.Tensor scalar or int

    Returns:
        padded_seqs: torch.Tensor of shape (B, T)
        labels: torch.Tensor of shape (B,)
        lengths: torch.Tensor of original sequence lengths (B,)
    """
  
    sequences, labels = zip(*batch)   # unzip the batch list
    

    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

   
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0) # shape → (B, T) where T = max sequence length in batch

   
    labels = torch.tensor(labels, dtype=torch.long)

   
    return padded_seqs, labels, lengths

   
#Step 2: Create DataLoaders for training/validation/testing

dataset = SMSDataset(df["text"], df["label"])

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
B=32
train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=B, shuffle=False, collate_fn=collate_fn)



# -------------------------------
# DEBUG: Check one batch of data
# x, y, lengths = next(iter(train_loader))  # or x, y if your collate_fn returns only padded_seqs and labels
# print("Input batch shape:", x.shape)
# print("Labels batch shape:", y.shape)
# print("Sequence lengths:", lengths)
# print("Labels in this batch:", y)

# # Optional: check for invalid labels
# assert y.min() >= 0 and y.max() < 2, "Labels are out of bounds!"







#Step 3: Define the training loop
# 1. Device setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 2. Model, loss function, optimizer
loss_func = nn.CrossEntropyLoss()
vocab_size = len(dataset.vocab)
model = RNNPadded(vocab_size=vocab_size, emb_dim=32, hidden_dim=64, output_dim=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

first_model =train_network(model = model, train_loader =train_loader, test_loader = test_loader, optimizer = optimizer, loss_func = loss_func, device = device, epochs=10 ,score_funcs={'Accuracy': accuracy_score}, checkpoint_file="rnn_padded.pth")
