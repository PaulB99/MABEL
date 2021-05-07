import unittest
import sys
sys.path.insert(0, '../')
import model, train
import torch
from transformers import BertTokenizer
import torch.nn as nn
import torch.optim as optim

# Test miniseq2seq functions
class TestMiniSeq2Seq(unittest.TestCase):
     
    # Check the model is set up correctly
    def test_setup(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mod = model.seq2seq(device, 7630)
        self.assertEqual(mod.vocab_size, 7630)
        self.assertEqual(device, mod.device)

    def test_tokeniser(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        tok=BertTokenizer("bert-base-uncased-vocab-mini.txt")
        string = 'Hello world'
        input_tensor = tok.encode(string, return_tensors="pt")[0].to(device)
        output = tok.decode(input_tensor)
        output = output.replace('[CLS] ', '')
        output = output.replace(' [SEP]', '')
        self.assertEqual(string, output)

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