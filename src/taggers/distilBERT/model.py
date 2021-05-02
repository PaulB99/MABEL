import torch.nn as nn
from transformers import DistilBertForSequenceClassification

# A basic BERT model for classification on the WNC dataset
class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "distilbert-base-uncased"   # I'd rather use a smaller one for testing, but this is the smallest that Transformers offers
        self.encoder = DistilBertForSequenceClassification.from_pretrained(options_name)

    def forward(self, label, text):
        loss, f = self.encoder(text, labels=label)[:2]
        return loss, f