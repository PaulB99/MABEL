import torch
import model
import time
from torchtext.data import Field, TabularDataset, BucketIterator
import torch.nn as nn
from transformers import DistilBertTokenizer
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import re
folder = re.split("\\\\|/",os.getcwd())[-5]
if folder=='pab734':
    os.environ['TORCH_HOME'] = '/data/labfs/pab734/torch'
    print('True')

data_path = '../../../data/'

# Tokeniser
tokeniser = DistilBertTokenizer.from_pretrained('distilbert-base-uncased') # Would rather use smaller for testing but this is the smallest Transformers offers

# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model parameters
MAX_SEQ_LEN = 128
PAD_INDEX = tokeniser.convert_tokens_to_ids(tokeniser.pad_token)
UNK_INDEX = tokeniser.convert_tokens_to_ids(tokeniser.unk_token)

# Fields
label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(use_vocab=False, tokenize=tokeniser.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
fields = [('label', label_field), ('text', text_field)]

# Dataset TODO: Work out how the data is coming in
train, valid = TabularDataset.splits(path=data_path, train='datasets/main/train_detection.csv', validation='datasets/main/valid_detection.csv',
                                           format='CSV', fields=fields, skip_header=True)

# Iterators
train_iter = BucketIterator(train, batch_size=32, sort_key=lambda x: len(x.text),
                            device=device, train=True, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid, batch_size=32, sort_key=lambda x: len(x.text),
                            device=device, train=True, sort=True, sort_within_batch=True)


# Training Function
def train(model,
          optimiser,
          criterion = nn.BCELoss(),
          train_loader = train_iter,
          valid_loader = valid_iter,
          num_epochs = 15,
          eval_every = len(train_iter) // 2,
          path = data_path,
          best_valid_loss = float("Inf")):
    
    # Initialise running values
    training_loss = 0.0
    valid_loss = 0.0
    global_step = 0
    
    # Arrays to store outputs
    training_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    
    print("Initialised")

    # Training loop
    start_time = time.clock()
    model.train()
    for epoch in range(num_epochs):
        for (labels, text), _ in train_loader:
            labels = labels.type(torch.LongTensor) # Biased or not    
            labels = labels.to(device)
            text = text.type(torch.LongTensor) # The text
            text = text.to(device)
            output = model(labels, text)
            loss, _ = output

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Update running values
            training_loss += loss.item()
            global_step += 1

            # Evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    

                    # Validation loop
                    for (labels, text), _ in valid_loader:
                        labels = labels.type(torch.LongTensor)           
                        labels = labels.to(device)
                        text = text.type(torch.LongTensor)  
                        text = text.to(device)
                        output = model(labels, text)
                        loss, _ = output
                        
                        valid_loss += loss.item()

                # Evaluate
                average_train_loss = training_loss / eval_every
                average_valid_loss = valid_loss / len(valid_loader)
                training_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # Reset running values
                testing_loss = 0.0                
                valid_loss = 0.0
                model.train()

                # Print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
                
    # TODO: Save output!
    state = {'training_loss': training_loss_list,
                  'validation_loss': valid_loss_list,
                  'steps': global_steps_list}
    
    # Save loss graph
    torch.save(state, ('../../../output/taggers/dist_training.pt'))
    plt.plot(global_steps_list, training_loss_list)
    plt.xlabel('Training steps')
    plt.ylabel('Training loss')
    plt.title('Training loss of distilbert model')
    plt.savefig('distilbert_loss.png')
    # Save model
    torch.save(model.state_dict(), '../../../cache/taggers/distilbert.pt')
    end_time = time.clock()
    train_time = end_time - start_time
    print('distilBERT model training complete in {} seconds'.format(train_time))
    print('Done!')
    
# Run the training
if __name__ == "__main__":
    mymodel = model.BERT().to(device)
    optimiser = optim.Adam(mymodel.parameters(), lr=2e-5)
    train(model=mymodel, optimiser=optimiser)
    
    