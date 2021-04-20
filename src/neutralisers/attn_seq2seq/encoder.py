import torch.nn as nn
import torch

# Encoder RNN for the seq2seq model

class EncoderRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        
        # Size of hidden layers
        self.hidden_size = hidden_size
        # Word embedding
        self.embedding = nn.Embedding(input_size, hidden_size)
        # GRU layer
        self.gru = nn.GRU(hidden_size, hidden_size)
        # Device to run on
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Forward function
    def forward(self, input, hidden):
        embed = self.embedding(input).view(1, 1, -1)
        output = embed
        # Apply GRU
        output, hidden = self.gru(output, hidden)
        return output, hidden

    # Initialise hidden layer
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)