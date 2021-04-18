import torch.nn as nn
import torch
import torch.nn.functional as f

# Encoder RNN for the seq2seq model

class DecoderRNN(nn.Module):
    
    def __init__(self, output_size, hidden_size):
        super(DecoderRNN, self).__init__()
        # Hidden layer size
        self.hidden_size = hidden_size
        # Word embeddings
        self.embedding = nn.Embedding(output_size, hidden_size)
        # GRU layer
        self.gru = nn.GRU(hidden_size, hidden_size)
        # Output layer
        self.out = nn.Linear(hidden_size, output_size)
        # Softmax layer
        self.softmax = nn.LogSoftmax(dim=1)
        # Device to run on
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Forward function
    def forward(self, input, hidden):
        # Apply layers in order
        output = self.embedding(input).view(1,1, -1) #1,1,-1
        output = f.relu(output)
        output, hidden = self.gru(output, hidden)
        #print(self.out(output[0]).size())
        output = self.softmax(self.out(output[0]))
        return output, hidden
    

    # Initialise hidden layers
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)