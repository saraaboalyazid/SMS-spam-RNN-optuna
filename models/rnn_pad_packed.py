# imports 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import sys, os
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from framework.core.rnn_utils import LastTimeStep  
from framework.core.rnn_utils import EmbeddingPackable
#el model
class RNNPadPacked(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, output_size, 
                 rnn_layers=1, bidirectional=False, dropout=0.0):
        super(RNNPadPacked, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional
        
        # Embedding layer wrapped to support PackedSequence
        self.embedding = EmbeddingPackable(nn.Embedding(vocab_size, emb_size, padding_idx=0))
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.LastTimeStep = LastTimeStep(rnn_layers=rnn_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, lengths=None):
        """
        x: torch.Tensor or PackedSequence
        lengths: required if x is tensor and variable length sequences
        """
        # If input is not PackedSequence, pack it first
        if not isinstance(x, PackedSequence):
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Embed (EmbeddingPackable handles PackedSequence)
        x = self.embedding(x)
        
        # RNN forward
        rnn_out, h_n = self.rnn(x)
        
        # Extract last relevant hidden state
        last = self.LastTimeStep((rnn_out, h_n))
        last = self.dropout(last)
        
        
        logits = self.fc(last)
        return logits


