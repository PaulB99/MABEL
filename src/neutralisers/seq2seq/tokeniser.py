import torch
import pandas as pd

# A tokeniser for the seq2seq model

class tokeniser():
    
    # Processes the columns of a dataframe
    def process_col(self, df):
            
        # Make lower case
        txt = df['text'].str.lower()
        
        # Remove punctuation and encode as ascii
        txt = txt.str.replace('[^A-Za-z\s]+', '')
        txt = txt.str.normalize('NFD')
        txt = txt.str.encode('ascii', errors='ignore').str.decode('utf-8')
        
        # Make lower case
        tar = df['text'].str.lower()
        
        # Remove punctuation and encode as ascii
        tar = tar.str.replace('[^A-Za-z\s]+', '')
        tar = tar.str.normalize('NFD')
        tar = tar.str.encode('ascii', errors='ignore').str.decode('utf-8')
     
        frame = { 'text': txt, 'target': tar }
        out = pd.DataFrame(frame)
        
        return out
        
    
    def init_lang(self, data_path):
        lang = {"SOS": 0, "EOS": 1}
        data_path = '../../../data/datasets/main/train_neutralisation.csv'
        data_df = pd.read_csv(data_path, header=None, names=['text', 'target'])
        count = 2
        
        txt = self.process_col(data_df)
        
        for index, row in txt.iterrows():
            sent = row['text'] + row['target']
            
            # Split and add to language
            words = sent.split(' ')
            for word in words:
                if lang.get(word) == None:
                    lang[word] = count
                    count+=1
        return lang
            
        
    def __init__(self, device, data_path):
        # Start and end of sequence tokens
        self.SOS_token = 0
        self.EOS_token = 1
        self.device = device
        self.lang = self.init_lang(data_path)
        self.lang_size = len(self.lang)
        
        
    def tokenise(self, sentence):
        # Make lower case
        sentence = sentence.str.lower()
        
        # Remove punctuation and encode as ascii
        sentence = sentence.str.replace('[^A-Za-z\s]+', '')
        sentence = sentence.str.normalize('NFD')
        sentence = sentence.str.encode('ascii', errors='ignore').str.decode('utf-8')
        words = sentence.split(' ')
        
        tokenised = []
        for word in words:
            tok = self.lang.get(word)
            tokenised.append(tok)
        
        
        return tokenised
    
    # Turn sentence into tensors
    def tensorise(self,sentence, lang):  
        indices = [lang.word2index[word] for word in sentence.split(' ')]
        indices.append(self.EOS_token)
        return torch.tensor(indices, dtype=torch.long, device=self.device).view(-1, 1)
    
    
    