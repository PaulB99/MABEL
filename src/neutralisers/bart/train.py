from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    BartConfig,
    Trainer,
    TrainingArguments,
    DataCollator,
    AutoTokenizer,
    )
from datasets import load_dataset
import torch

# Training script for bart based neutraliser

data_path = '../../../data/'
output_path = '../../../output'

# Load datasets
train_dataset = load_dataset('csv', data_files=data_path+'datasets/main/train_neutralisation.csv')
valid_dataset = load_dataset('csv', data_files=data_path+'datasets/main/valid_neutralisation.csv')

# Tokeniser
#tokeniser = BartTokenizer.from_pretrained('facebook/bart-large') 
tokeniser = AutoTokenizer.from_pretrained("facebook/bart-base")

# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model configuration
config = BartConfig()

model = BartForConditionalGeneration(config=config)

# Data collator
data_collator = DataCollator(
    tokenizer = tokeniser, model=model)

# Arguments for trainer
train_args = TrainingArguments(
    output_dir = output_path,
    overwrite_output_dir = True,
    num_train_epochs = 6,
    per_gpu_batch_size = 64, 
    save_steps = 10_000,
    save_total_limit = 2
    )

# Huggingface trainer to train the model
trainer = Trainer(
    model = model,
    args = train_args,
    data_collator = data_collator,
    train_dataset = train_dataset,
    eval_dataset = valid_dataset,
    tokenizer = tokeniser,
    )


