import torch
import torch.nn as nn
import model as md
import torch.optim as optim
import pandas as pd
import tokeniser
from transformers import BertTokenizer
import time
import matplotlib.pyplot as plt
import os

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_step(model, input_tensors, target_tensors, optimiser, criterion):
    optimiser.zero_grad()

    loss = 0
    epoch_loss = 0
    for x in range(len(input_tensors)):
        input_tensor = input_tensors[x]
        target_tensor = target_tensors[x]

        output = model(input_tensor, target_tensor)
        
        # Calculate the loss from prediction and target
        target_length = target_tensor.shape[0]
        for index in range(target_length):
            loss += criterion(output[index], target_tensor[index])

    loss.backward()
    optimiser.step()
    epoch_loss = loss.item()

    return epoch_loss

# Load in data
data_path = '../../../data/datasets/main/train_neutralisation.csv'
data_df = pd.read_csv(data_path, header=None, skiprows=1, names=['text', 'target'])

# Batch size
batch_size = 64

#tok = tokeniser.tokeniser(device, data_path)
tok = BertTokenizer.from_pretrained('bert-base-uncased') 
lang_size = 30522

print('Tokeniser initialised of size {}'.format(lang_size))
model = md.seq2seq(device, lang_size).to(device)
model.train()
print('Model initialised')

optimiser = optim.SGD(model.parameters(), lr=0.00003)
criterion = nn.NLLLoss()
total_loss_iterations = 0
num_epochs = 11

start_time = time.perf_counter()
loss_vals = []
loss_points = []

for i in range(num_epochs): #num_epochs
    j=0
    input_back = []
    target_back = []
    for index, row in data_df.iterrows():
        input_tensor = tok.encode(row['text'], return_tensors="pt")[0].to(device)
        input_tensor = input_tensor.view(-1, 1)
        target_tensor = tok.encode(row['target'], return_tensors="pt")[0].to(device)
        target_tensor = target_tensor.view(-1, 1)
        #input_tensor = tok.tensorise(row['text'])
        #target_tensor = tok.tensorise(row['target'])
        if j % batch_size == 0:
            input_back.append(input_tensor)
            target_back.append(target_tensor)
            loss = train_step(model, input_back, target_back, optimiser, criterion)
            total_loss_iterations += loss
            input_back = []
            target_back = []
        else:
            input_back.append(input_tensor)
            target_back.append(target_tensor)
    
        if j % 5000 == 0:
            average_loss= total_loss_iterations / 5000
            if j != 0:
                loss_vals.append(average_loss)
                loss_points.append(j)
            total_loss_iterations = 0
            print('%d %.4f' % (j, average_loss))
        j+=1
        
    if os.path.exists('../../../cache/neutralisers/seq2seq.pt'):  # checking if there is a file with this name
        os.remove('../../../cache/neutralisers/seq2seq.pt')
    torch.save(model.state_dict(), '../../../cache/neutralisers/seq2seq.pt')
    print('Model saved after epoch {}'.format(str(i)))
      
end_time = time.perf_counter()   
print('Seq2seq model trained in {}'.format(end_time-start_time)) 
torch.save(model.state_dict(), '../../../cache/neutralisers/seq2seq.pt')
print('Model saved!')

# Save loss graph
plt.plot(loss_points, loss_vals)
plt.xlabel('Training steps')
plt.ylabel('Training loss')
plt.title('Training loss of seq2seq model')
plt.savefig('loss_graph.png')    
