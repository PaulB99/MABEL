import torch.nn as nn
import torch
import torch.nn.functional as f

# Encoder RNN with attention for the seq2seq model

class DecoderRNN(nn.Module):
    
    def __init__(self, hidden_size, output_size, dropout_per=0.1, max_length=64):
        super(DecoderRNN, self).__init__()
        # Hidden layer size
        self.hidden_size = hidden_size
        # Output layer size
        self.output_size = output_size
         # Device to run on
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Max sequence length
        self.max_length = max_length
        # Word embeddings
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # Attention layers
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_comb = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # Dropout change
        self.dropout_per = dropout_per
        # Droput layer
        self.dropout = nn.Dropout(self.dropout_per)
        # GRU layer
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        # Output layer
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # Create embeddings and dropout
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # Apply attention layer
        attn_weights = f.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_comb(output).unsqueeze(0)

        # Apply relu and GRU
        output = f.relu(output)
        output, hidden = self.gru(output, hidden)

        # Softmax and output
        output = f.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)