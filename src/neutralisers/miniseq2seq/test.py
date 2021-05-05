from transformers import (
    BertTokenizer,
    )
import torch
import pandas as pd
import nltk
import model

data_path = '../../../data/'
output_path = '../../../output'

# Tokeniser
tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model parameters
MAX_SEQ_LEN = 128

# Helper function to load model checkpoint
def load_ckpt(load_path, model):
    model.load_state_dict(torch.load(load_path))

def test(model):
    running_score = 0
    counter = 0
    correct = 0
    test_data = pd.read_csv(data_path+'datasets/main/test_neutralisation.csv')
    #test_data = load_dataset('csv', data_files=data_path+'datasets/main/test_neutralisation.csv')
    for index, row in test_data.iterrows():
        text = [row['text']]
        full_preds = []
        if len(text[0]) < 128 and len(row['target']) < 128:
            counter+=1
            for s in text:
                pred=model.generate(s, tokeniser)
                pred = pred.replace('<s>', '')
                pred = pred.replace('</s>', '')
                pred_list = pred.split(' ')
                full_preds+=pred_list
            split_target = row['target'].split(' ')
            score = nltk.translate.bleu_score.sentence_bleu([split_target], full_preds)
            if split_target == full_preds:
                counter+=1
            running_score+=score
            if counter%1000 == 0:
                print(len(text))
                print(len(text[0]))
                print(full_preds)
                print(split_target)
                print('For BLEU : {}'.format(score))
    final_score = running_score/counter
    accuracy = correct/counter
    print('Bleu score ' + str(final_score) + ' over {} examples'.format(counter))
    print('Accuracy ' + str(accuracy))
    
    
if __name__ == "__main__":
    neutraliser_path = '../../../cache/neutralisers/miniseq2seq.pt'
    mymodel = model.seq2seq(device, 7630).to(device)
    load_ckpt(neutraliser_path, mymodel)
    test(mymodel)
