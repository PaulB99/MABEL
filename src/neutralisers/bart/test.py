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
from datasets import load_dataset, load_metric
import numpy as np
from torchtext.data.metrics import bleu_score
import pandas as pd
import nltk

data_path = '../../../data/'
output_path = '../../../output'

# Tokeniser
tokeniser = BartTokenizer.from_pretrained("facebook/bart-base")

# Metrics
metric = load_metric("sacrebleu")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model parameters
MAX_SEQ_LEN = 256

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

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

# Postprocess text for evaluation
def postpro(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

# Calculate sacrebleu score
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokeniser.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokeniser.batch_decode(labels, skip_special_tokens=True)

    # Simple post-processing
    decoded_preds, decoded_labels = postpro(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokeniser.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def test(model):
    running_score = 0
    counter=0
    test_data = pd.read_csv(data_path+'datasets/main/test_neutralisation.csv')
    #test_data = load_dataset('csv', data_files=data_path+'datasets/main/test_neutralisation.csv')
    for index, row in test_data.iterrows():
        counter+=1
        s = row['text']
        inp = tokeniser([s], max_length=256, return_tensors='pt').to(device)
        pred_tensors=model.generate(inp['input_ids']).to(device)
        pred_list = [tokeniser.decode(p) for p in pred_tensors].split(' ')
        split_target = row['target'].split(' ')
        score = nltk.translate.bleu_score.sentence_bleu([split_target], pred_list)
        running_score+=score
        if counter%1000 == 0:
            print(pred_list)
            print(split_target)
            print('For BLEU : {}'.format(score))
    final_score = running_score/counter
    print('Bleu score ' + str(final_score) + ' over {} examples'.format(counter))
    
    
if __name__ == "__main__":
    config = BartConfig(d_model=512,
                            encoder_layers=6,
                            decoder_layers=6,
                            encoder_attention_heads=8,
                            decoder_attention_heads=8,
                            encoder_ffn_dim=2048,
                            decoder_ffn_dim=2048,
                            activation_function='gelu'
                            )
    mymodel = BartForConditionalGeneration(config=config).from_pretrained('../../../cache/neutralisers/bart').to(device)
    #mymodel = model.BART().to(device)
    #optimiser = optim.Adam(mymodel.parameters(), lr=2e-5)
    #train(model=mymodel, optimiser=optimiser)
    test(mymodel)
