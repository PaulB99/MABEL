import torch 
import sys
import time
sys.path.insert(0, '../')
from taggers.base_model import model as t_model
from neutralisers.bart import model as n_model
from transformers import BertTokenizer, BartTokenizer
import transformers
from torchtext.data import Field, Dataset, BucketIterator, Example

transformers.logging.set_verbosity_error()

# Program to load pretrained models and run the full pipeline

class runner():
    
    # Helper function to load model checkpoint
    def load_ckpt(self,load_path, model):
        model.load_state_dict(torch.load(load_path))
    
    def initialise(self):
         # Device to run on
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Tokenisers
        # Select right tokenisers
        if self.tagger=='base_model':   
            self.t_tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
        elif self.tagger=='large_model':
            self.t_tokeniser = BertTokenizer.from_pretrained('bert-large-uncased')
        else:
            raise ValueError('Model {} not recognised'.format(self.tagger))
        
        if self.neutraliser=='bart':
            self.n_tokeniser = BartTokenizer.from_pretrained("facebook/bart-base")
        else:
            raise ValueError('Model {} not recognised'.format(self.tagger))
        
        # Padding
        MAX_SEQ_LEN = 128
        PAD_INDEX = self.t_tokeniser.convert_tokens_to_ids(self.t_tokeniser.pad_token)
        UNK_INDEX = self.t_tokeniser.convert_tokens_to_ids(self.t_tokeniser.unk_token)
        
        # Fields
        label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
        text_field = Field(use_vocab=False, tokenize=self.t_tokeniser.encode, lower=False, include_lengths=False, batch_first=True, fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
        self.fields = [('labels', label_field), ('text', text_field)]
        print('Initialised')
        
        # Load models
        t1 = time.clock()
        tagger_path = '../../cache/taggers/' + self.tagger + '.pt'
        self.tagger_model = t_model.BERT().to(self.device)
        self.load_ckpt(tagger_path, self.tagger_model)
        t2 = time.clock()
        print('Tagger loaded in {}s!'.format(t2-t1))
        
        t1 = time.clock()
        neutraliser_path = '../../cache/neutralisers/' + self.neutraliser + '.pt'
        self.neutraliser_model = n_model.BART().to(self.device)
        self.load_ckpt(neutraliser_path, self.neutraliser_model)
        t2 = time.clock()
        print('Neutraliser loaded in {}s!'.format(t2-t1))
    
    def __init__(self, tagger, neutraliser):
        self.tagger = tagger
        self.neutraliser = neutraliser
        self.initialise()
    
    # The full detection and neutralisation pipeline
    def pipeline(self, sentence):
        
        # Split sentences
        split_sent = sentence.split('.')
            
        # Populate dataset
        prov_data = []
        for x in split_sent:
            ex = Example.fromlist([0,x], self.fields)
            prov_data.append(ex)
        sent_data = Dataset(prov_data, self.fields)    
            
        iterator = BucketIterator(sent_data, batch_size=1, device=self.device, train=False, shuffle=False, sort=False, sort_key=lambda x: len(x.text))
            
        # Output vals
        output_array = []
        biased_indices = []
        ticker = -1
            
        # Check for bias
        with torch.no_grad():
            for (labels, text), _ in iterator:
                ticker+=1
                labels = labels.type(torch.LongTensor)           
                labels = labels.to(self.device)
                text = text.type(torch.LongTensor)  
                text = text.to(self.device)
                tagger_output = self.tagger_model(labels, text)
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
        PAD_INDEX = self.n_tokeniser.convert_tokens_to_ids(self.n_tokeniser.pad_token)
        UNK_INDEX = self.n_tokeniser.convert_tokens_to_ids(self.n_tokeniser.unk_token)
            
        # Fields
        text_field = Field(use_vocab=False, tokenize=self.n_tokeniser.encode, lower=False, include_lengths=False, batch_first=True, fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
        fields = [('text', text_field)]
            
        # Populate dataset
        prov_data = []
        for x in split_sent:
            ex = Example.fromlist([x], fields)
            prov_data.append(ex)
        sent_data = Dataset(prov_data, fields)
            
        iterator = BucketIterator(sent_data, batch_size=1, device=self.device, train=False, shuffle=False, sort=False, sort_key=lambda x: len(x.text))
            
        # Neutralise sentences tagged as biased
        ticker=-1
        with torch.no_grad():
            for (text), _ in iterator:
                ticker+=1
                if ticker in biased_indices:
                    text = text.type(torch.LongTensor)  
                    text = text.to(self.device)
                    neutraliser_output = self.neutraliser_model.generate(text)
                    decoded = self.n_tokeniser.decode(neutraliser_output[0], skip_special_tokens=True)
                    output_array.insert(ticker, decoded)
    
        output_string = ''
        for s in output_array:
            output_string+=(s+'. ')
        return output_string
           
            
# Run
if __name__ == "__main__":
    args = str(sys.argv)
    r = runner('base_model', 'bart')
    sentence = str(input('Enter the phrase to be neutralised, or Exit to quit\n'))
    while not sentence == 'Exit':
        print(r.pipeline(sentence))
        sentence = str(input('Enter the phrase to be neutralised, or Exit to quit\n'))

    