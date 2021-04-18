import torch 
import sys
import time
sys.path.insert(0, '../')
from taggers.base_model import model as base_model
from taggers.large_model import model as large_model
from taggers.lexi import model as lexi_model
from neutralisers.bart import model as bart_model
from neutralisers.roberta import model as roberta_model
from neutralisers.parrot import model as parrot_model
from neutralisers.seq2seq import model as seq2seq_model
from transformers import BertTokenizer, BartTokenizer
import transformers
from torchtext.data import Field, Dataset, BucketIterator, Example
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
import pandas as pd

transformers.logging.set_verbosity_error()

# Program to load pretrained models and run the full pipeline

class runner():
    
    # Helper function to load model checkpoint
    def load_ckpt(self,load_path, model):
        model.load_state_dict(torch.load(load_path))
    
    def initialise(self):
         # Device to run on
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Start loading
        t1 = time.perf_counter()
        tagger_path = '../../cache/taggers/' + self.tagger + '.pt'
        
        self.bert = False
        # Select right tokenisers
        if self.tagger=='base_model':   
            self.t_tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
            self.tagger_model = base_model.BERT().to(self.device)
            self.bert = True
        elif self.tagger=='large_model':
            self.t_tokeniser = BertTokenizer.from_pretrained('bert-large-uncased')
            self.tagger_model = large_model.BERT().to(self.device)
            self.bert = True
        elif self.tagger=='lexi':
            self.tagger_model = lexi_model.lexi()
            t2 = time.perf_counter()
            print('Tagger loaded in {}s!'.format(t2-t1))
            
        else:
            raise ValueError('Model {} not recognised'.format(self.tagger))
    
        if self.bert:
            # Padding
            MAX_SEQ_LEN = 128
            PAD_INDEX = self.t_tokeniser.convert_tokens_to_ids(self.t_tokeniser.pad_token)
            UNK_INDEX = self.t_tokeniser.convert_tokens_to_ids(self.t_tokeniser.unk_token)
            
            # Fields
            label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
            text_field = Field(use_vocab=False, tokenize=self.t_tokeniser.encode, lower=False, include_lengths=False, batch_first=True, fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
            self.fields = [('labels', label_field), ('text', text_field)]
            
            # Load tagger model
            self.load_ckpt(tagger_path, self.tagger_model)
            t2 = time.perf_counter()
            print('Tagger loaded in {}s!'.format(t2-t1))
            
        
        # Neutralisers
        t1 = time.perf_counter()
        if self.neutraliser=='bart':
            self.n_tokeniser = BartTokenizer.from_pretrained("facebook/bart-base")
            
        elif self.neutraliser=='seq2seq':
            self.n_tokeniser=BertTokenizer.from_pretrained('bert-base-uncased')
            neutraliser_path = '../../cache/neutralisers/' + self.neutraliser + '.pt'
            self.neutraliser_model = seq2seq_model.seq2seq(self.device, 30522).to(self.device)
            self.load_ckpt(neutraliser_path, self.neutraliser_model)
            t2 = time.perf_counter()
            print('Neutraliser loaded in {}s!'.format(t2-t1))
            return
        
        elif self.neutraliser=='parrot':
            self.neutraliser_model = parrot_model.parrot()
            t2 = time.perf_counter()
            print('Neutraliser loaded in {}s!'.format(t2-t1))
            return
            
        # Roberta is loaded in a different way
        elif self.neutraliser=='roberta':
            model_args = Seq2SeqArgs()
            model_args.num_train_epochs = 11
            model_args.evaluate_generated_text = True
            model_args.evaluate_during_training = True
            
            self.neutraliser_model = Seq2SeqModel(
                "roberta",
                encoder_decoder_name="../../cache/neutralisers/roberta",
                args=model_args,
                )
            
            t2 = time.perf_counter()
            print('Neutraliser loaded in {}s!'.format(t2-t1))
            
            # Break out early - we don't need a tokeniser and stuff for this one
            return
        else:
            raise ValueError('Model {} not recognised'.format(self.tagger))
         
        neutraliser_path = '../../cache/neutralisers/' + self.neutraliser + '.pt'
        self.neutraliser_model = bart_model.BART().to(self.device)
        self.load_ckpt(neutraliser_path, self.neutraliser_model)
        t2 = time.perf_counter()
        print('Neutraliser loaded in {}s!'.format(t2-t1))
    
    def __init__(self, tagger, neutraliser):
        self.tagger = tagger
        self.neutraliser = neutraliser
        self.initialise()
    
    # The full detection and neutralisation pipeline
    def pipeline(self, sentence):
        
        biased_bool=False
        
        # Split sentences
        split_sent = sentence.split('.')
        
        # Lexi is easier
        if not self.bert:
            output_array = []
            biased_indices = []
            ticker = -1
            
            for s in split_sent:
                ticker+=1
                biased = self.tagger_model.predict(s)
                if biased == 1:
                    biased_bool=True
                    #print('Biased!')
                    biased_indices.append(ticker)
                elif biased == 0:
                    #print('Unbiased!')
                    output_array.insert(ticker, split_sent[ticker])
                
        
        # Bert uses a different pipeline
        else:
            
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
                        biased_bool=True
                        #print('Biased!')
                        biased_indices.append(ticker)
                    elif biased == 0:
                        #print('Unbiased!')
                        output_array.insert(ticker, split_sent[ticker])
                    
        # TIME TO NEUTRALISE!
        
        # The parrot pipeline is greatly simplified
        if self.neutraliser=='parrot':
            ticker = -1
            for s in split_sent:
                ticker+=1
                if ticker in biased_indices:
                    pred=self.neutraliser_model.generate(s)
                    output_array.insert(ticker, pred)
                    
        # Using the roberta neutraliser has a different pipeline
        elif self.neutraliser=='roberta':
            ticker = -1
            for s in split_sent:
                ticker+=1
                if ticker in biased_indices:
                    pred=self.neutraliser_model.predict(s)
                    output_array.insert(ticker, pred)
                    
        elif self.neutraliser=='seq2seq':
            ticker = -1
            for s in split_sent:
                ticker+=1
                if ticker in biased_indices:
                    pred=self.neutraliser_model.generate(s)
                    output_array.insert(ticker, pred)
                
        else:
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
            output_string+=(s+'.')
        return output_string, biased_bool
           