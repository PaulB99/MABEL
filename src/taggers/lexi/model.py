import os
# A very simple lightweight tagger model, specifically for weaker machines

class lexi():
    
    def __init__(self):
        file = open('../../data/bias_lex/bias-lexicon.txt', 'r')
        lines = file.readlines()
        self.words = []
        for line in lines:
            line.strip()
            self.words.append(line)


    # Return true if in bias lexicon, false otherwise
    def predict(self,sentence):
        split_sent = sentence.split(' ')
        for s in split_sent:
            if s in self.words:
                return 1
        return 0
    

    
#x = lexi()
#cwd = os.getcwd()  # Get the current working directory (cwd)
#files = os.listdir(cwd)
#print(files)

                