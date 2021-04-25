from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    BartConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    Trainer
    )
import torch
import time
import model
from torchtext.data import Field, TabularDataset, BucketIterator
import torch.nn as nn
import torch.optim as optim

data_path = '../../../data/'
output_path = '../../../output'

# Tokeniser
tokeniser = BartTokenizer.from_pretrained("facebook/bart-base")

# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model parameters
MAX_SEQ_LEN = 128
PAD_INDEX = tokeniser.convert_tokens_to_ids(tokeniser.pad_token)
UNK_INDEX = tokeniser.convert_tokens_to_ids(tokeniser.unk_token)

# Fields
text_field = Field(use_vocab=False, tokenize=tokeniser.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
target_field = Field(use_vocab=False, tokenize=tokeniser.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
fields = [('text', text_field), ('target', target_field)]

# Dataset TODO: Work out how the data is coming in
train, valid = TabularDataset.splits(path=data_path, train='datasets/main/train_neutralisation.csv', validation='datasets/main/valid_neutralisation.csv',
                                           format='CSV', fields=fields, skip_header=True)

# Iterators
train_iter = BucketIterator(train, batch_size=8, sort_key=lambda x: len(x.text),
                            device=device, train=True, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid, batch_size=8, sort_key=lambda x: len(x.text),
                            device=device, train=True, sort=True, sort_within_batch=True)


# Training Function
def train(model,
          optimiser,
          criterion = nn.BCELoss(),
          train_loader = train_iter,
          valid_loader = valid_iter,
          num_epochs = 11,
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
        for (text, target), _ in train_loader:
            text = text.type(torch.LongTensor) # Biased or not    
            text = text.to(device)
            target = target.type(torch.LongTensor) # The text
            target = target.to(device)
            output = model(text, target)
            loss, _ = output

            optimiser.zero_grad()
            loss.mean().backward()
            optimiser.step()

            # Update running values
            training_loss += loss.mean().item()
            global_step += 1

            # Evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    

                    # Validation loop
                    for (text, target), _ in valid_loader:
                        text = text.type(torch.LongTensor)           
                        text = text.to(device)
                        target = target.type(torch.LongTensor)  
                        target = target.to(device)
                        output = model(text, target)
                        loss, _ = output
                        
                        valid_loss += loss.mean().item()

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
    
    torch.save(state, ('../../../output/neutralisers/bart_training.pt'))
    # Save model
    torch.save(model.state_dict(), '../../../cache/neutralisers/bart.pt')
    end_time = time.clock()
    train_time = end_time - start_time
    print('Bart training complete in {} seconds'.format(train_time))
    print('Done!')
  
def alt_train(model):
    training_args = Seq2SeqTrainingArguments(
        output_dir='../../../cache/neutralisers/bart',          
        num_train_epochs=11,           
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=8,   
        warmup_steps=500,               
        weight_decay=0.01,
        learning_rate=0.003,
        )
    
    trainer = Seq2SeqTrainer(
    model=model,                       
    args=training_args,                  
    train_dataset=train,        
    eval_dataset=valid,
)
    trainer.train()
    
# Run the training
if __name__ == "__main__":
    mymodel = model.BART().to(device)
    #optimiser = optim.Adam(mymodel.parameters(), lr=2e-5)
    #train(model=mymodel, optimiser=optimiser)
    alt_train(mymodel)
    