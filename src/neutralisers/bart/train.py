from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    BartConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
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
tokeniser = BartTokenizer.from_pretrained("facebook/bart-base")

# Label dataset
#train_dataset = tokeniser.encode(train_dataset_unlabeled, add_special_tokens=True, return_tensors='pt')
#valid_dataset = tokeniser.encode(valid_dataset_unlabeled, add_special_tokens=True, return_tensors='pt')

# Preprocess dataset 
train_in = train_dataset['train']['input']
train_ta = train_dataset['train']['target']

train_inputs = tokeniser(train_in, truncation=True)
train_targets = tokeniser(train_ta, truncation=True)
print(type(train_inputs))

train_inputs['labels'] = train_targets["input_ids"]
print(type(train_inputs))

train_dataset = train_dataset.map(
            train_inputs,
            batched=True,
            #num_proc=data_args.preprocessing_num_workers,
            #remove_columns=column_names,
            #load_from_cache_file=not data_args.overwrite_cache,
        )


# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model configuration
config = BartConfig()

model = BartForConditionalGeneration(config=config)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer = tokeniser)

# Arguments for trainer
train_args = TrainingArguments(
    output_dir = output_path,
    overwrite_output_dir = True,
    num_train_epochs = 6,
    per_device_train_batch_size = 64, 
    save_steps = 10_000,
    save_total_limit = 2
    )

# Huggingface trainer to train the model
trainer = Trainer(
    model = model,
    args = train_args,
    data_collator = data_collator,
    train_dataset = train_dataset,
    #train_dataset = train_inputs,
    eval_dataset = valid_dataset,
    tokenizer = tokeniser,
    )


result = trainer.train()
trainer.save_model('../../../cache/bart_neut')

metrics = result.metrics
trainer.log_metrics('train', metrics)
trainer.save_metrics('train', metrics)
trainer.save_state()



