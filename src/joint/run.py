import torch 
import sys
import time
sys.path.insert(0, '../')
from taggers.base_model import model as t_model
from neutralisers.bart import model as n_model
from transformers import BertTokenizer, BartTokenizer
from torchtext.data import Field, Dataset, BucketIterator


# Program to load pretrained models and run the full pipeline

# Helper function to load model checkpoint
def load_ckpt(load_path, model):
    model.load_state_dict(torch.load(load_path))
    #print(f'Trained model loaded from <== {load_path}')

# 
def pipeline(sentence, tagger='base_model', neutraliser='bart'):
    
    # Split sentences
    split_sent = sentence.split('.')
    
    # Populate dataset
    prov_data = [('labels','text')]
    for x in split_sent:
        prov_data.append(('0', x))
    
    # Device to run on
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Tokenisers
    t_tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
    n_tokeniser = BartTokenizer.from_pretrained("facebook/bart-base")
    
    # Padding
    MAX_SEQ_LEN = 128
    PAD_INDEX = t_tokeniser.convert_tokens_to_ids(t_tokeniser.pad_token)
    UNK_INDEX = t_tokeniser.convert_tokens_to_ids(t_tokeniser.unk_token)
    
    # Fields
    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(use_vocab=False, tokenize=t_tokeniser.encode, lower=False, include_lengths=False, batch_first=True, fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
    fields = [('labels', label_field), ('text', text_field)]
    sent_data = Dataset(prov_data, fields)
    print('Data initialised')
    
    iterator = BucketIterator(sent_data, batch_size=1, device=device, train=False, shuffle=False, sort=False)
    print('Iterator initialised')
    
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
    
    with torch.no_grad():
        for (labels, text), _ in iterator:
            labels = labels.type(torch.LongTensor)           
            labels = labels.to(device)
            text = text.type(torch.LongTensor)  
            text = text.to(device)
            biased = t_model(labels, text)
            print(biased)
            if biased == 1:
                print('Good')

# Run
pipeline('Example')
    