import torch 
import sys
import time
sys.path.insert(0, '../')
from taggers.base_model import model as t_model
from neutralisers.bart import model as n_model
from transformers import BertTokenizer, BartTokenizer
from torchtext.data import Field, Dataset, BucketIterator, Example

# Program to load pretrained models and run the full pipeline

# Helper function to load model checkpoint
def load_ckpt(load_path, model):
    model.load_state_dict(torch.load(load_path))

# The full detection and neutralisation pipeline
def pipeline(sentence, tagger='base_model', neutraliser='bart'):
    
    # Split sentences
    split_sent = sentence.split('.')
    
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
    
    # Populate dataset
    prov_data = []
    for x in split_sent:
        ex = Example.fromlist([0,x], fields)
        prov_data.append(ex)
    sent_data = Dataset(prov_data, fields)
        
    print('Data initialised')
    
    iterator = BucketIterator(sent_data, batch_size=1, device=device, train=False, shuffle=False, sort=False, sort_key=lambda x: len(x.text))
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
    
    # Output vals
    output_array = []
    biased_indices = []
    ticker = -1
    
    # Check for bias
    with torch.no_grad():
        for (labels, text), _ in iterator:
            ticker+=1
            labels = labels.type(torch.LongTensor)           
            labels = labels.to(device)
            text = text.type(torch.LongTensor)  
            text = text.to(device)
            tagger_output = tagger_model(labels, text)
            _,tagger_output = tagger_output
            biased = torch.argmax(tagger_output, 1)
            if biased == 1:
                print('Biased!')
                biased_indices.append(ticker)
            elif biased == 0:
                print('Unbiased!')
                output_array.insert(ticker, split_sent[ticker])
                
                
    # Prepare data for neutralisation
    # Padding
    MAX_SEQ_LEN = 128
    PAD_INDEX = n_tokeniser.convert_tokens_to_ids(n_tokeniser.pad_token)
    UNK_INDEX = n_tokeniser.convert_tokens_to_ids(n_tokeniser.unk_token)
    
    # Fields
    text_field = Field(use_vocab=False, tokenize=n_tokeniser.encode, lower=False, include_lengths=False, batch_first=True, fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
    fields = [('text', text_field)]
    
    # Populate dataset
    prov_data = []
    for x in split_sent:
        ex = Example.fromlist([x], fields)
        prov_data.append(ex)
    sent_data = Dataset(prov_data, fields)
    
    iterator = BucketIterator(sent_data, batch_size=1, device=device, train=False, shuffle=False, sort=False, sort_key=lambda x: len(x.text))
    print('Ready for neutralisation')
    
    # Neutralise sentences tagged as biased
    ticker=-1
    with torch.no_grad():
        for (text), _ in iterator:
            ticker+=1
            text = text.type(torch.LongTensor)  
            text = text.to(device)
            neutraliser_output = neutraliser_model.generate(text)
            output_array.insert(ticker, neutraliser_output)

    return output_array

# Run
if __name__ == "__main__":
    args = str(sys.argv)
    print(pipeline("Rushden is the worst town"))
    