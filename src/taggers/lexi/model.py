import os
import string
# A very simple lightweight tagger model, specifically for weaker machines

class lexi():
    
    def __init__(self):
        file = open('../../data/bias_lex/bias-lexicon.txt', 'r')
        lines = file.readlines()
        self.words = []
        for line in lines:
            line = line.strip()
            self.words.append(line)


    # Return true if in bias lexicon, false otherwise
    def predict(self,sentence):
        split_sent = sentence.split(' ')
        for s in split_sent:
            # Lowercase and strip punctuation
            s = s.lower()
            s = s.translate(str.maketrans('', '', string.punctuation))
            if s in self.words:
                return 1
        return 0
    
