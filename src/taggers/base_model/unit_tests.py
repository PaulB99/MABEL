import unittest
import sys
sys.path.insert(0, '../')
import model, train
import torch
from transformers import BertTokenizer
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, Dataset, BucketIterator, Example
from transformers import BertTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import os
import re
import torch.optim as optim

def load_ckpt(load_path, model):
    #state_dict = torch.load(load_path, map_location=device)
    #model.load_state_dict(state_dict['model_state_dict'])
    model.load_state_dict(torch.load(load_path))
    print(f'Trained model loaded from <== {load_path}')
    #return state_dict['valid_loss']

# Test miniseq2seq functions
class TestBERT(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        data_path = '../../../data/'
    
        # Tokeniser
        cls.tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Check if GPU is available
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters
        PAD_INDEX = cls.tokeniser.convert_tokens_to_ids(cls.tokeniser.pad_token)
        UNK_INDEX = cls.tokeniser.convert_tokens_to_ids(cls.tokeniser.unk_token)
        
        # Fields
        label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
        text_field = Field(use_vocab=False, tokenize=cls.tokeniser.encode, lower=False, include_lengths=False, batch_first=True, fix_length=128, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
        fields = [('label', label_field), ('text', text_field)]
        
        prov_data = []
        examples = ['Hello world'] * 10
        for x in examples:
            ex = Example.fromlist([0,x], fields)
            prov_data.append(ex)
        test_data = Dataset(prov_data, fields)    
        # Test data iterator
        cls.test_iter = BucketIterator(test_data, batch_size=16, device=device, train=False, shuffle=False, sort=False, sort_key=lambda x: len(x.text))
        
        cls.mymodel = model.BERT().to(device)
    
    # Check the tokeniser works correctly
    def test_tokeniser(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        string = 'hello world'
        input_tensor = TestBERT.tokeniser.encode(string, return_tensors="pt")[0].to(device)
        output = TestBERT.tokeniser.decode(input_tensor)
        output = output.replace('[CLS] ', '')
        output = output.replace(' [SEP]', '')

        self.assertEqual(string, output)

    # Check if weights change with training
    def test_training(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        before_param = TestBERT.mymodel.parameters()
        optimiser = optim.Adam(TestBERT.mymodel.parameters(), lr=0.01)
        TestBERT.mymodel.train()
        for (labels, text), _ in TestBERT.test_iter:
            labels = labels.type(torch.LongTensor) # Biased or not    
            labels = labels.to(device)
            text = text.type(torch.LongTensor) # The text
            text = text.to(device)
            output = TestBERT.mymodel(labels, text)
            loss, _ = output

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        after_param = TestBERT.mymodel.parameters
        
        self.assertNotEqual(before_param, after_param)
        
    # Check if checkpoints can be loaded correctly
    def test_load(self):
        before_param = TestBERT.mymodel.parameters()
        load_ckpt('../../../cache/taggers/base_model.pt', TestBERT.mymodel)
        after_param = TestBERT.mymodel.parameters()
        
        self.assertNotEqual(before_param, after_param)
        

if __name__ == '__main__':
    
    unittest.main(verbosity=2)