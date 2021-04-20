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
SOS_token = 101
EOS_token = 102

class seq2seq(nn.Module):
    
    def __init__(self, device, input_size):
        
       super().__init__()
      
        # Initialise the encoder and decoder
       self.encoder = EncoderRNN(input_size, 64)
       self.decoder = DecoderRNN(input_size, 64)
       self.vocab_size = input_size
       self.device = device       
      
     # Forward function
    def forward(self, text, target, teacher_forcing_ratio=0.5, max_length=128):
        
        encoder_hidden = self.encoder.initHidden()

        # Get input size
        input_length = text.size(0) #0
        batch_size = 1 #target.shape[1] #1
        target_length = target.size(0) #0
          
        # Initialise variable for predictions
        #encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size, device=self.device)
    
        # Encode each word
        for i in range(input_length):
            encoder_output, encoder_hidden = self.encoder(text[i], encoder_hidden)
            #encoder_outputs[i] = encoder_output[0,0]
    
        # Use the encoder’s hidden layer as the decoder's hidden
        decoder_hidden = encoder_hidden
      
        # Add start of sequence token
        decoder_input = torch.tensor([SOS_token], device=self.device)
        #decoder_input= torch.tensor(device=self.device)
    
        outputs = torch.zeros(target_length, 1, self.vocab_size).to(self.device)
        
        # For each word, run decoder and make predictions
        for t in range(target_length):   
            
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            
            # If teacher forcing, next decoder input is the next word. If not, use the highest value from decoder
            teacher_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1)
            #decoder_input = topi
            decoder_input = (target[t] if teacher_force else topi.squeeze())
            if(teacher_force == False and decoder_input.item() == EOS_token):
                break
        return outputs

    
    # Neutralise a given sequence
    def generate(self, sentence, tokeniser):
        with torch.no_grad():
            
            encoder_hidden = self.encoder.initHidden()

            
            # Tokenise and make into tensor
            input_tensor = tokeniser.encode(sentence, return_tensors="pt")[0].to(self.device)

            # Follow a path similar to forward
            input_length = input_tensor.size(0)
            target_length = input_length
             # Initialise variable for predictions
            outputs = torch.zeros(target_length, 1, self.vocab_size).to(self.device)
            
            # Encode each word
            for i in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[i], encoder_hidden)
                
            # Use the encoder’s hidden layer as the decoder's hidden
            decoder_hidden = encoder_hidden
          
            # Add start of sequence token
            decoder_input = torch.tensor([SOS_token], device=self.device)
            #decoder_input= torch.tensor(device=self.device)
        
            # For each word, run decoder and make predictions
            for t in range(target_length):   
                
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[t] = decoder_output
                
                topv, topi = decoder_output.topk(1)
                decoder_input = topi
                if decoder_input.item() == EOS_token:
                    break
                
            decoded_words = []
      
      
            for ot in range(outputs.size(0)):
                topv, topi = outputs[ot].topk(1)
                decoded_words.append(tokeniser.decode([topi[0].item()]))
                
        decoded_sentence = ' '.join(decoded_words)
        return decoded_sentence