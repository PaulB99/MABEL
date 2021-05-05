import unittest
import sys
sys.path.insert(0, '../')
import model, train
import torch
from transformers import BertTokenizer
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
from transformers import BertTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import os
import re

def load_ckpt(load_path, model):
    #state_dict = torch.load(load_path, map_location=device)
    #model.load_state_dict(state_dict['model_state_dict'])
    model.load_state_dict(torch.load(load_path))
    print(f'Trained model loaded from <== {load_path}')
    #return state_dict['valid_loss']

# Test miniseq2seq functions
class TestBERT(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls) -> None:
        data_path = '../../../data/'
    
        # Tokeniser
        tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Check if GPU is available
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters
        PAD_INDEX = tokeniser.convert_tokens_to_ids(tokeniser.pad_token)
        UNK_INDEX = tokeniser.convert_tokens_to_ids(tokeniser.unk_token)
        
        # Fields
        label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
        text_field = Field(use_vocab=False, tokenize=tokeniser.encode, lower=False, include_lengths=False, batch_first=True, fix_length=128, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
        fields = [('label', label_field), ('text', text_field)]
        
        # Load in data 
        test_data = TabularDataset(path=data_path+'datasets/main/test_detection.csv',format='CSV', fields=fields, skip_header=True)
        
        # Test data iterator
        test_iter = BucketIterator(test_data, batch_size=16, device=device, train=False, shuffle=False, sort=False, sort_key=lambda x: len(x.text))
        
        cls.mymodel = model.BERT().to(device)
        load_ckpt('../../../cache/taggers/base_model.pt', cls.mymodel)
    
     
    # Check the model is set up correctly
    def test_setup(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.assertEqual(device, TestBERT.mymodel.device)

    def test_tokeniser(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        tok=BertTokenizer("bert-base-uncased-vocab-mini.txt")
        string = 'Hello world'
        input_tensor = tok.encode(string, return_tensors="pt")[0].to(device)
        output_tensor = tok.decode(input_tensor)
        self.assertEqual(input_tensor, output_tensor)

    def test_loader(self):
        
    def test_flow(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        tok=BertTokenizer("bert-base-uncased-vocab-mini.txt")
        mod = model.seq2seq(device, 7630)
        output1 = mod.generate('Hello world', tok)
        #self.assertEqual(output1, 'Hello world')
        input_tensor = tok.encode('Hello world', return_tensors="pt")[0].to(device)
        input_tensor = input_tensor.view(-1, 1)
        target_tensor = tok.encode('Hello world', return_tensors="pt")[0].to(device)
        target_tensor = input_tensor.view(-1, 1)
        optimiser = optim.SGD(mod.parameters(), lr=0.01)
        criterion = nn.NLLLoss()

        for i in range(10):
            loss = train.train_step(mod, input_tensor, target_tensor, optimiser, criterion)
        output2 = mod.generate('Hello world', tok)
        self.assertNotEqual(output1, output2)
        

if __name__ == '__main__':
    
    unittest.main()