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
from datasets import load_dataset

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
train_a, valid_a = TabularDataset.splits(path=data_path, train='datasets/main/train_neutralisation.csv', validation='datasets/main/valid_neutralisation.csv',
                                           format='CSV', fields=fields, skip_header=True)

# Iterators
train_iter = BucketIterator(train_a, batch_size=8, sort_key=lambda x: len(x.text),
                            device=device, train=True, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid_a, batch_size=8, sort_key=lambda x: len(x.text),
                            device=device, train=True, sort=True, sort_within_batch=True)

# Preprocess data for bart model                            
def prepro(examples):
        inputs = [ex for ex in examples['text']]
        targets = [ex for ex in examples['target']]
        #inputs = examples['text']
        #targets = examples['target']
        #inputs = [inp for inp in inputs]
        model_inputs = tokeniser(inputs, max_length=MAX_SEQ_LEN, padding=False, truncation=True)

        # Setup the tokenizer for targets
        with tokeniser.as_target_tokenizer():
            labels = tokeniser(targets, max_length=MAX_SEQ_LEN, padding=False, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


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
    
    # Prepare data
    train = load_dataset('csv', data_files=data_path+'datasets/main/train_neutralisation.csv')
    train = train["train"]
    cols = train.column_names
    print(cols)
    train = train.map(
            prepro,
            batched=True,
            num_proc=None,
            remove_columns=cols,
        )
    
    
    valid = load_dataset('csv', data_files=data_path+'datasets/main/valid_neutralisation.csv')
    #print(valid['train'])
    valid = valid["train"]
    valid = valid.map(
            prepro,
            batched=True,
            num_proc=None,
            remove_columns=cols,
        )
    
    
    label_pad_token_id = tokeniser.pad_token_id
    
    training_args = Seq2SeqTrainingArguments(
        output_dir='../../../cache/neutralisers/bart',          
        num_train_epochs=11,           
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=8,   
        warmup_steps=500,               
        weight_decay=0.01,
        learning_rate=0.003,
        )
    
    data_collator = DataCollatorForSeq2Seq(
            tokeniser,
            model=model,
            label_pad_token_id=label_pad_token_id,
            )
    
    trainer = Seq2SeqTrainer(
        model=model,                       
        args=training_args,                  
        train_dataset=train,        
        eval_dataset=valid,
        tokenizer = tokeniser,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model()
    
# Run the training
if __name__ == "__main__":
    mymodel = model.BART().to(device)
    #optimiser = optim.Adam(mymodel.parameters(), lr=2e-5)
    #train(model=mymodel, optimiser=optimiser)
    alt_train(mymodel)
    