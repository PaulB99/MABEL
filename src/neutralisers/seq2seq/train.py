import torch
import torch.nn as nn
import model as md
import torch.optim as optim
import pandas as pd
import tokeniser

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_step(model, input_tensor, target_tensor, optimiser, criterion):
    optimiser.zero_grad()

    input_length = input_tensor.size(0)
    loss = 0
    epoch_loss = 0

    output = model(input_tensor, target_tensor)

    num_iter = output.size(0)

    # Calculate the loss from prediction and target
    for x in range(num_iter):
        loss += criterion(output[x], target_tensor[x])

    loss.backward()
    optimiser.step()
    epoch_loss = loss.item() / num_iter

    return epoch_loss

# Load in data
data_path = '../../../data/datasets/main/train_neutralisation.csv'
data_df = pd.read_csv(data_path, header=None, names=['text', 'target'])

tok = tokeniser.tokeniser(device, data_path)
print('Tokeniser initialised')
model = md.seq2seq(device, tok.lang_size)
print('Model initialised')

optimiser = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()
total_loss_iterations = 0
num_epochs = 11

for i in range(num_epochs):
    for j in range():
        training_pair = data_df[iter - 1]
        input_tensor = tok.tokenise(training_pair[0])
        target_tensor = tok.tokenise(training_pair[1])
    
        loss = train_step(model, input_tensor, target_tensor, optimiser, criterion)
    
        total_loss_iterations += loss
    
        if iter % 5000 == 0:
            avarage_loss= total_loss_iterations / 5000
            total_loss_iterations = 0
            print('%d %.4f' % (iter, avarage_loss))
          
torch.save(model.state_dict(), '../../../cache/neutralisers/seq2seq.pt')