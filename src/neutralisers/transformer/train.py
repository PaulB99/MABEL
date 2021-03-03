# Train the transformers based neutralisation model
import torch
from torchtext.data import Field, TabularDataset, BucketIterator
import torch.nn as nn
from transformers import RobertaTokenizer, Trainer, TrainingArguments, RobertaConfig, DataCollatorForLanguageModeling,
import torch.optim as optim

data_path = '../../../data/'
output_path = '../../../output'

# Tokeniser
tokeniser = RobertaTokenizer.from_pretrained('roberta-base') # Would rather use smaller for testing but this is the smallest Transformers offers

# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model configuration
config = RobertaConfig()

model = RobertaForMaskedLM(config=config)

dataset = 

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokeniser, mlm=True, mlm_probability=0.15)

# Arguments for trainer
train_args = TrainingArguments(
    output_dir = output_path,
    overwrite_output_dir = True,
    num_train_epochs = 6,
    per_gpu_batch_size = 64, 
    save_steps = 10_000
    save_total_limit = 2)

# Huggingface trainer to train the model
trainer = Trainer(
    model = model,
    args = train_args,
    data_collator = data_collator,
    train_dataset = dataset,
    prediction_loss_only = True
    )

# Train!
trainer.train()
trainer.save_model(output_path)

