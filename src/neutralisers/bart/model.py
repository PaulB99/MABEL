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
        
        config = BartConfig(d_model=512,
                            encoder_layers=6,
                            decoder_layers=6,
                            encoder_attention_heads=8,
                            decoder_attention_heads=8,
                            encoder_ffn_dim=2048,
                            decoder_ffn_dim=2048,
                            activation_function='gelu'
                            )
        self.encoder = BartForConditionalGeneration(config=config)

    def forward(self, text):
        loss, f = self.encoder(text)[:2]
        return loss, f
    
    def generate(self, text):
        output = self.encoder.generate(text)
        return output