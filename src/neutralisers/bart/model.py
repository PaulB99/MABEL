import torch.nn as nn
from transformers import (
    BartForConditionalGeneration,
    BartConfig,
    )

# A basic BART model for neutralisation on the WNC dataset
class BART(nn.Module):

    def __init__(self):
        super(BART, self).__init__()
        
        options_name = "bert-large-uncased"
        
        config = BartConfig()
        self.encoder = BartForConditionalGeneration(config=config)

    def forward(self, label, text):
        loss, f = self.encoder(text, labels=label)[:2]
        return loss, f