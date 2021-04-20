import os
import string
# A very simple lightweight neutraliser model, specifically for weaker machines

class lexi():
    
    def __init__(self):
        file = open('../../data/bias_lex/bias-lexicon.txt', 'r')
        lines = file.readlines()
        self.words = []
        for line in lines:
            line = line.strip()
            self.words.append(line)
        file.close()


    # Return true if in bias lexicon, false otherwise
    def generate(self,sentence):
        split_sent = sentence.split(' ')
        new_sent = []
        for s in split_sent:
            # Lowercase and strip punctuation
            s = s.lower()
            s = s.translate(str.maketrans('', '', string.punctuation))
            if s not in self.words:
                new_sent.append(s)
        sent = " ".join(str(x) for x in new_sent)
        
        return sent
    