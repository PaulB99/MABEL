import unittest
import sys
sys.path.insert(0, '../')
import model, train
import torch
from transformers import BertTokenizer

# Test miniseq2seq functions
class TestMiniSeq2Seq(unittest.TestCase):
     
    # Check the model is set up correctly
    '''
    def test_setup(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mod = model.seq2seq(device, 7630)
        self.assertEqual(mod.vocab_size, 7630)
        self.assertEqual(device, mod.device)
        '''

    def test_tokeniser(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        tok=BertTokenizer("bert-base-uncased-vocab-mini.txt")
        input_tensor = tok.encode('Hello word', return_tensors="pt")[0].to(device)
        output_tensor = tok.decode(input_tensor)
        self.assertEqual(input_tensor, output_tensor)


if __name__ == '__main__':
    
    unittest.main()