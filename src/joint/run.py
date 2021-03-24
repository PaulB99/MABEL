import torch 
import sys
import time
sys.path.insert(0, '../')
from taggers.base_model import model as t_model
from neutralisers.bart import model as n_model
from transformers import BertTokenizer, BartTokenizer


# Program to load pretrained models and run the full pipeline

# Helper function to load model checkpoint
def load_ckpt(load_path, model):
    model.load_state_dict(torch.load(load_path))
    print(f'Trained model loaded from <== {load_path}')

# 
def pipeline(sentence, tagger='base_model', neutraliser='bart'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    t_tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
    n_tokeniser = BartTokenizer.from_pretrained("facebook/bart-base")
    
    # Load models
    t1 = time.clock()
    tagger_path = '../../cache/taggers/' + tagger + '.pt'
    tagger_model = t_model.BERT().to(device)
    load_ckpt(tagger_path, tagger_model)
    t2 = time.clock()
    print('Tagger loaded in {}s!'.format(t2-t1))
    
    t1 = time.clock()
    neutraliser_path = '../../cache/neutralisers/' + neutraliser + '.pt'
    neutraliser_model = n_model.BART().to(device)
    load_ckpt(neutraliser_path, neutraliser_model)
    t2 = time.clock()
    print('Neutraliser loaded in {}s!'.format(t2-t1))
    
    t_tok = t_tokeniser(sentence)
    biased = tagger_model(0,t_tok)
    
    if biased == 1:
        print('Good')

# Run
pipeline('Example')
    