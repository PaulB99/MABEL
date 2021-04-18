import sys
sys.path.insert(0, '../')
import joint.run as run
from simple_term_menu import TerminalMenu
# Run the command line app

if __name__ == "__main__":
    acceptable_taggers = ['base_model','large_model', 'lexi']
    acceptable_neutralisers = ['bart','roberta','parrot', 'seq2seq']
    
    print('Welcome to the bias neutralisation program! Select a tagger model below')
    
    tagger_menu = TerminalMenu(acceptable_taggers)
    neutraliser_menu = TerminalMenu(acceptable_neutralisers)
    
    tagger = acceptable_taggers[tagger_menu.show()]
    
    print('Please select a neutraliser model')
    neu = acceptable_neutralisers[neutraliser_menu.show()]
        
    #try:
    r = run.runner(tagger, neu)
    #except:
        #print('There was a problem. Make sure you have trained the models you want to use!')
        #sys.exit()
        
    sentence = str(input('Enter the phrase to be neutralised, or Exit to quit\n'))
    while not sentence == 'Exit':
        output, biased = r.pipeline(sentence)
        #if biased:
         #   print('Biased!')
        #else:
         #   print('Unbiased!')
        print(output)
        sentence = str(input('Enter the phrase to be neutralised, or Exit to quit\n'))