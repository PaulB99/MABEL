import torch.nn as nn
from transformers import (
    BartForConditionalGeneration,
    BartConfig,
    )

# A small BART model for neutralisation on the WNC dataset
# Half the number of layers, heads and ffn size
class BART(nn.Module):

    def __init__(self):
        super(BART, self).__init__()
        
        config = BartConfig(d_model=1024,
                            encoder_layers=12,
                            decoder_layers=12,
                            encoder_attention_heads=16,
                            decoder_attention_heads=16,
                            encoder_ffn_dim=2048,
                            decoder_ffn_dim=2048,
                            activation_function='gelu'
                            )
        self.encoder = BartForConditionalGeneration(config=config)

    def forward(self, text, target):
        loss, f = self.encoder(text, target)[:2]
        return loss, f
    
    def generate(self, text):
        output = self.encoder.generate(text)
        return output