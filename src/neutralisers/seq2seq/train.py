import torch
import torch.nn as nn
import model as md
import torch.optim as optim
import pandas as pd
import tokeniser
from transformers import BertTokenizer

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_step(model, input_tensor, target_tensor, optimiser, criterion):
    optimiser.zero_grad()

    #input_length = input_tensor.size(0)
    loss = 0
    epoch_loss = 0

    output = model(input_tensor, target_tensor)
    
    # Remove spurious 1 dimension
    output = torch.squeeze(output)
    
    # Calculate the loss from prediction and target
    loss += criterion(output, target_tensor)

    loss.backward()
    optimiser.step()
    epoch_loss = loss.item()

    return epoch_loss

# Load in data
data_path = '../../../data/datasets/main/train_neutralisation.csv'
data_df = pd.read_csv(data_path, header=None, skiprows=1, names=['text', 'target'])

#tok = tokeniser.tokeniser(device, data_path)
tok = BertTokenizer.from_pretrained('bert-base-uncased') 
lang_size = 30522

print('Tokeniser initialised of size {}'.format(lang_size))
model = md.seq2seq(device, lang_size).to(device)
model.train()
print('Model initialised')

optimiser = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()
total_loss_iterations = 0
num_epochs = 11

for i in range(num_epochs):
    j=0
    for index, row in data_df.iterrows():
        input_tensor = tok.encode(row['text'], return_tensors="pt")[0]
        target_tensor = tok.encode(row['target'], return_tensors="pt")[0]
        #input_tensor = tok.tensorise(row['text'])
        #target_tensor = tok.tensorise(row['target'])
    
        loss = train_step(model, input_tensor, target_tensor, optimiser, criterion)
    
        total_loss_iterations += loss
    
        if j % 5000 == 0:
            average_loss= total_loss_iterations / 5000
            total_loss_iterations = 0
            print('%d %.4f' % (j, average_loss))
        j+=1
          
torch.save(model.state_dict(), '../../../cache/neutralisers/seq2seq.pt')