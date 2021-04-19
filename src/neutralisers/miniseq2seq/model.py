import torch.nn as nn
import torch
import os
import re
folder = re.split("\\\\|/",os.getcwd())[-1]
if folder=='miniseq2seq':
    from encoder import EncoderRNN
    from decoder import DecoderRNN
else:
    from .encoder import EncoderRNN
    from .decoder import DecoderRNN
import random

# The seq2seq model

# Start and end of sequence tokens
SOS_token = 0
#EOS_token = 1

class seq2seq(nn.Module):
    
    def __init__(self, device, input_size):
        
       super().__init__()
      
        # Initialise the encoder and decoder
       self.encoder = EncoderRNN(input_size, 64)
       self.decoder = DecoderRNN(input_size, 64)
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
            decoder_input = topi
            #input = (target[t] if teacher_force else topi)
            #if(teacher_force == False and input.item() == EOS_token):
             #   break
        return outputs

    
    # Neutralise a given sequence
    def generate(self, sentence, tokeniser):
        with torch.no_grad():
            
            # Tokenise and make into tensor
            input_tensor = tokeniser.encode(sentence, return_tensors="pt")[0].to(self.device)

            # Follow a path similar to forward
            input_length = input_tensor.size(0)
            target_length = input_length
             # Initialise variable for predictions
            outputs = torch.zeros(target_length, 1, self.vocab_size).to(self.device)
            
            # Encode each word
            for i in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[i])
    
            # Use the encoder’s hidden layer as the decoder's hidden
            decoder_hidden = encoder_hidden.to(self.device)
          
            # Add start of sequence token
            decoder_input = torch.tensor([SOS_token], device=self.device)
            #decoder_input= torch.tensor(device=self.device)
        
            # For each word, run decoder and make predictions
            for t in range(target_length):   
                
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[t] = decoder_output
                
                topv, topi = decoder_output.topk(1)
                decoder_input = topi
                
            decoded_words = []
      
      
            for ot in range(outputs.size(0)):
                topv, topi = outputs[ot].topk(1)
                decoded_words.append(tokeniser.decode([topi[0].item()]))
        return decoded_words