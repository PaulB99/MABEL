import torch.nn as nn
import torch
import encoder as en
import decoder as de
import random

# The seq2seq model

# Start and end of sequence tokens
SOS_token = 0
#EOS_token = 1

class seq2seq(nn.Module):
    
    def __init__(self, device, input_size):
        
       super().__init__()
      
        # Initialise the encoder and decoder
       self.encoder = en.EncoderRNN(input_size, 256)
       self.decoder = de.DecoderRNN(input_size, 256)
       self.vocab_size = input_size
       self.device = device       
      
     # Forward function
    def forward(self, text, target, teacher_forcing_ratio=1):

        # Get input size
        input_length = text.size(0) #0
        batch_size = 1 #target.shape[1] #1
        target_length = target.shape[0] #0
          
        # Initialise variable for predictions
        outputs = torch.zeros(target_length, batch_size, self.vocab_size).to(self.device)
    
        # Encode each word
        for i in range(input_length):
            encoder_output, encoder_hidden = self.encoder(text[i])
    
        # Use the encoder’s hidden layer as the decoder's hidden
        decoder_hidden = encoder_hidden.to(self.device)
      
        # Add start of sequence token
        decoder_input = torch.tensor([SOS_token], device=self.device)
        #decoder_input= torch.tensor(device=self.device)
    
        # For each word, run decoder and make predictions
        for t in range(target_length):   
            
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            
            # If teacher forcing, next decoder input is the next word. If not, use the highest value from decoder
            teacher_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1)
            #input = topi
            #input = (target[t] if teacher_force else topi)
            #if(teacher_force == False and input.item() == EOS_token):
             #   break
        return outputs

    
    # Neutralise a given sequence
    def generate(self, sentence):
        with torch.no_grad():
            input_tensor = tensorFromSentence(input_lang, sentences[0])
            output_tensor = tensorFromSentence(output_lang, sentences[1])
      
            decoded_words = []
      
            output = model(input_tensor, output_tensor)
            # print(output_tensor)
      
            for ot in range(output.size(0)):
                topv, topi = output[ot].topk(1)
                # print(topi)
    
                if topi[0].item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(output_lang.index2word[topi[0].item()])
        return decoded_words