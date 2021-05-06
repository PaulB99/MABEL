from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    BartConfig,
    )
import torch
import pandas as pd
import nltk

data_path = '../../../data/'
output_path = '../../../output'

# Tokeniser
tokeniser = BartTokenizer.from_pretrained("facebook/bart-base")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model parameters
MAX_SEQ_LEN = 128

# Get the number of words changed between 2 samples
def worddiff(a, b):
    a_list = a.split(' ')
    b_list = b.split(' ')
    diff = 0
    for i in a_list:
        if i not in b_list:
            diff+=1
    for j in b_list:
        if j not in a_list:
            diff+=1
    return diff
    

def test(model):
    running_score = 0
    counter = 0
    correct = 0
    test_data = pd.read_csv(data_path+'datasets/main/test_neutralisation.csv')
    #test_data = load_dataset('csv', data_files=data_path+'datasets/main/test_neutralisation.csv')
    for index, row in test_data.iterrows():
        text = [row['text']]
        full_preds = []
        if len(text[0]) < 120 and len(row['target']) < 120:
            counter+=1
            for s in text:
                inp = tokeniser([s], max_length=128, return_tensors='pt').to(device)
                pred_tensors=model.generate(inp['input_ids']).to(device)
                pred_list = [tokeniser.decode(p) for p in pred_tensors]
                pred_list = pred_list[0].replace('<s>', '')
                pred_list = pred_list.replace('</s>', '')
                pred_list = pred_list.split(' ')
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
 

def onedifftest(model):
    running_score = 0
    counter = 0
    correct = 0
    test_data = pd.read_csv(data_path+'datasets/main/test_neutralisation.csv')
    #test_data = load_dataset('csv', data_files=data_path+'datasets/main/test_neutralisation.csv')
    for index, row in test_data.iterrows():
        text = [row['text']]
        full_preds = []
        if len(text[0]) < 128 and len(row['target']) < 128:
            if worddiff(text[0], row['target']) == 1:
                counter+=1
                for s in text:
                    inp = tokeniser([s], max_length=128, return_tensors='pt').to(device)
                    pred_tensors=model.generate(inp['input_ids']).to(device)
                    pred_list = [tokeniser.decode(p) for p in pred_tensors]
                    pred_list = pred_list[0].replace('<s>', '')
                    pred_list = pred_list.replace('</s>', '')
                    pred_list = pred_list.split(' ')
                    full_preds+=pred_list
                split_target = row['target'].split(' ')
                score = nltk.translate.bleu_score.sentence_bleu([split_target], full_preds)
                if split_target == full_preds:
                    counter+=1
                running_score+=score
                if counter%500 == 0:
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
    onedifftest(mymodel)
