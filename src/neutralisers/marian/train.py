from simpletransformers.seq2seq import Seq2SeqModel

# A neutraliser using the marian framework

model = Seq2SeqModel(
    encoder_decoder_type="marian",
    encoder_decoder_name="Helsinki-NLP/opus-mt-en-de",
)
