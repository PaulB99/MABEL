from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs


# A seq2seq model with a roBERTa encoder and a BERT decoder

class roberta():
    
    def __init__(self, model_args):
        self.model = Seq2SeqModel(
            "roberta",
            "roberta-base",
            "bert-base-cased",
            args=model_args
)
        