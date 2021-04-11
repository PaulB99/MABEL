from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
import pandas as pd
from model import roberta

# Training script for roberta model

data_path = '../../../data/'
output_path = '../../../output'

train_data = pd.read_csv(data_path+'datasets/main/train_neutralisation.csv')
valid_data = pd.read_csv(data_path+'datasets/main/valid_neutralisation.csv')

train_data.columns = ['input_text','target_text']
valid_data.columns = ['input_text','target_text']

model_args = Seq2SeqArgs()
model_args.num_train_epochs = 11
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
best_model_dir = '../../../cache/neutralisers/roberta'

mymodel = roberta(model_args)

mymodel.model.train_model(train_data, eval_data=valid_data, use_cuda=True, verbose=False)