from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
import torch


# A seq2seq model with a roBERTa encoder and a BERT decoder

class roberta():
    
    def __init__(self, model_args):
        
        cuda_available = torch.cuda.is_available()
        
        self.model = Seq2SeqModel(
            "roberta",
            "roberta-base",
            "bert-base-uncased",
            use_cuda=cuda_available,
            args=model_args
)
        