# imports
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from framework.core.rnn_utils import LastTimeStep


# el model
class RNNPadded(nn.Module):
    """
    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim):
        - define embedding layer (with padding_idx=0)
        - define RNN layer (use nn.RNN, nn.GRU, or nn.LSTM)
        - define fully connected layer (hidden_dim → output_dim)
    """

    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim):
        super(RNNPadded, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.RNN(emb_dim, hidden_dim, batch_first=True)
        self.LastTimeStep = LastTimeStep(rnn_layers=1, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, output_dim)

    # def forward(self, x):
    #     # (B, T)
    #     emb = self.embedding(x)        # → (B, T, D) , D = emb_dim
    #     out, _ = self.rnn(emb)         # → out: (B, T, H) → all hidden states for all time steps , _: (num_layers * num_directions, B, H) → the final hidden state (ignored here)
    #     last = self.LastTimeStep(out)          # → (B, H)
    #     logits = self.fc(last)         # → (B, C)
    #     return logits
    def forward(self, x):
        # print("Input x shape:", x.shape)  # debug

        emb = self.embedding(x)
        # print("Embedding shape:", emb.shape)  # debug

        out, h_n = self.rnn(emb)
        # print("RNN out shape:", out.shape)  # debug
        # print("RNN h_n shape:", h_n.shape)  # debug

        last = self.LastTimeStep((out, h_n))
        # print("LastTimeStep output shape:", last.shape)  # debug

        logits = self.fc(last)
        # print("Logits shape:", logits.shape)  # debug
        return logits
