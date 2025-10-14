import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from torch.nn.utils.rnn import PackedSequence
class LastTimeStep(nn.Module):
    def __init__(self, rnn_layers=1, bidirectional=False):
        super(LastTimeStep, self).__init__()
        self.rnn_layers = rnn_layers
        self.num_driections = 2 if bidirectional else 1

    def forward(self, input):
        rnn_output = input[0]
        last_step = input[1]

        if isinstance(last_step, tuple):  # LSTM case
            last_step = last_step[0]

        batch_size = last_step.shape[1]
       

        last_step = last_step.contiguous().view(self.rnn_layers, self.num_driections, batch_size, -1)
        
        last_step = last_step[self.rnn_layers-1]
       
        last_step = last_step.permute(1, 0, 2)
     
        last_step = last_step.reshape(batch_size, -1)
    
        return last_step
class LastTimeStep_debuged(nn.Module):
    def __init__(self, rnn_layers=1, bidirectional=False):
        super(LastTimeStep, self).__init__()
        self.rnn_layers = rnn_layers
        self.num_driections = 2 if bidirectional else 1

    def forward(self, input):
        rnn_output = input[0]
        last_step = input[1]

        if isinstance(last_step, tuple):  # LSTM case
            last_step = last_step[0]

        batch_size = last_step.shape[1]
        print("Original h_n shape:", last_step.shape)  # <-- debug line

        last_step = last_step.contiguous().view(self.rnn_layers, self.num_driections, batch_size, -1)
        print("After view:", last_step.shape)  # <-- debug line

        last_step = last_step[self.rnn_layers-1]
        print("After selecting last layer:", last_step.shape)  # <-- debug line

        last_step = last_step.permute(1, 0, 2)
        print("After permute:", last_step.shape)  # <-- debug line

        last_step = last_step.reshape(batch_size, -1)
        print("After final reshape (will go to Linear):", last_step.shape)  # <-- debug line

        return last_step

# class LastTimeStep(nn.Module):
#     """
#     A class for extracting the hidden activations of the last time step following 
#     the output of a PyTorch RNN module. 
#     """
#     def __init__(self, rnn_layers=1, bidirectional=False):
#         super(LastTimeStep, self).__init__()
#         self.rnn_layers = rnn_layers
#         if bidirectional:
#             self.num_driections = 2
#         else:
#             self.num_driections = 1    
    
#     def forward(self, input):
#         #Result is either a tupe (out, h_t)
#         #or a tuple (out, (h_t, c_t))
#         rnn_output = input[0]

#         last_step = input[1]
#         if(type(last_step) == tuple):
#             last_step = last_step[0]
#         batch_size = last_step.shape[1] #per docs, shape is: '(num_layers * num_directions, batch, hidden_size)'
        
#         last_step = last_step.contiguous().view(self.rnn_layers, self.num_driections, batch_size, -1)
#         #We want the last layer's results
#         last_step = last_step[self.rnn_layers-1] 
#         #Re order so batch comes first
#         last_step = last_step.permute(1, 0, 2)
#         #Finally, flatten the last two dimensions into one
#         return last_step.reshape(batch_size, -1)
    
class EmbeddingPackable(nn.Module):
    """
    The embedding layer in PyTorch does not support Packed Sequence objects. 
    This wrapper class will fix that. If a normal input comes in, it will 
    use the regular Embedding layer. Otherwise, it will work on the packed 
    sequence to return a new Packed sequence of the appropriate result. 
    """
    def __init__(self, embd_layer):
        super(EmbeddingPackable, self).__init__()
        self.embd_layer = embd_layer 
    
    def forward(self, input):
        if type(input) == torch.nn.utils.rnn.PackedSequence:
            # We need to unpack the input, 
            sequences, lengths = torch.nn.utils.rnn.pad_packed_sequence(input.cpu(), batch_first=True)
            #Embed it
            sequences = self.embd_layer(sequences.to(input.data.device))
            #And pack it into a new sequence
            return torch.nn.utils.rnn.pack_padded_sequence(sequences, lengths.cpu(), 
                                                           batch_first=True, enforce_sorted=False)
        else:#apply to normal data
            return self.embd_layer(input)

