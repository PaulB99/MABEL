import torch
import torch.nn as nn
import model as md
import torch.optim as optim
import pandas as pd
import tokeniser
from tokenizers import BertWordPieceTokenizer
import time
import matplotlib.pyplot as plt

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
tok = BertWordPieceTokenizer("bert-base-uncased-vocab-mini.txt", lowercase=True)
lang_size = 30522

print('Tokeniser initialised of size {}'.format(lang_size))
model = md.seq2seq(device, lang_size).to(device)
model.train()
print('Model initialised')

optimiser = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.NLLLoss()
total_loss_iterations = 0
num_epochs = 1

start_time = time.perf_counter()
loss_vals = []
loss_points = []

for i in range(num_epochs):
    j=0
    for index, row in data_df.iterrows():
        input_tensor = tok.encode(row['text'], return_tensors="pt")[0].to(device)
        target_tensor = tok.encode(row['target'], return_tensors="pt")[0].to(device)
        #input_tensor = tok.tensorise(row['text'])
        #target_tensor = tok.tensorise(row['target'])
    
        loss = train_step(model, input_tensor, target_tensor, optimiser, criterion)
    
        total_loss_iterations += loss
    
        if j % 500 == 0:
            average_loss= total_loss_iterations / 500
            if j != 0:
                loss_vals.append(average_loss)
                loss_points.append(j)
            total_loss_iterations = 0
            print('%d %.4f' % (j, average_loss))
        j+=1
  
end_time = time.perf_counter()   
print('Mini seq2seq model trained in {}'.format(end_time-start_time)) 
torch.save(model.state_dict(), '../../../cache/neutralisers/miniseq2seq.pt')
print('Model saved!')

# Save loss graph
plt.plot(loss_points, loss_vals)
plt.xlabel('Training steps')
plt.ylabel('Training loss')
plt.title('Training loss of miniseq2seq model')
plt.savefig('loss_graph.png')    
