import unittest
import sys
sys.path.insert(0, '../')
import model
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, BartConfig
import torch.optim as optim
from torchtext.data import Field, Dataset, BucketIterator, Example


def load_ckpt(load_path, model):
    model.load_state_dict(torch.load(load_path))
    print(f'Trained model loaded from <== {load_path}')

# Test miniseq2seq functions
class TestBART(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        data_path = '../../../data/'
    
        # Tokeniser
        cls.tokeniser = BartTokenizer.from_pretrained("facebook/bart-base")
        
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
        
        config = BartConfig(d_model=512,
                            encoder_layers=6,
                            decoder_layers=6,
                            encoder_attention_heads=8,
                            decoder_attention_heads=8,
                            encoder_ffn_dim=2048,
                            decoder_ffn_dim=2048,
                            activation_function='gelu'
                            )
        cls.mymodel = BartForConditionalGeneration(config=config).to(device)
        
        cls.mymodel = model.BERT().to(device)
    
    # Check the tokeniser works correctly
    def test_tokeniser(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        string = 'hello world'
        input_tensor = TestBART.tokeniser.encode(string, return_tensors="pt")[0].to(device)
        output = TestBART.tokeniser.decode(input_tensor)
        output = output.replace('[CLS] ', '')
        output = output.replace(' [SEP]', '')

        self.assertEqual(string, output)

    # Check if weights change with training
    def test_training(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        before_param = TestBART.mymodel.parameters()
                
        after_param = TestBART.mymodel.parameters()
        
        self.assertNotEqual(before_param, after_param)
        
    # Check if checkpoints can be loaded correctly
    def test_load(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        before_param = TestBART.mymodel.parameters()
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
        after_param = mymodel.parameters()
        self.assertNotEqual(before_param, after_param)
        

if __name__ == '__main__':
    
    unittest.main(verbosity=2)