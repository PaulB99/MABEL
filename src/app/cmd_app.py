import sys
sys.path.insert(0, '../')
import joint.run as run
from simple_term_menu import TerminalMenu
import transformers
transformers.logging.set_verbosity_error()

# Run the command line app

if __name__ == "__main__":
    acceptable_taggers = ['base_model','large_model', 'distilbert','lexi']
    acceptable_neutralisers = ['bart','roberta','parrot', 'seq2seq', 'miniseq2seq', 'lexi_swap']
    
    print('Welcome to the MABEL system! Select a tagger model below')
    
    tagger_menu = TerminalMenu(acceptable_taggers)
    neutraliser_menu = TerminalMenu(acceptable_neutralisers)
    
    tagger = acceptable_taggers[tagger_menu.show()]
    
    print('Please select a neutraliser model')
    neu = acceptable_neutralisers[neutraliser_menu.show()]
        
    try:
        r = run.runner(tagger, neu)
    except:
        print('There was a problem. Make sure you have trained the models you want to use!')
        sys.exit()
        
    sentence = str(input('Enter the phrase to be neutralised, or Exit to quit\n'))
    while not (sentence == 'Exit' or sentence == 'exit'):
        output, biased = r.pipeline(sentence)
        print(output)
        sentence = str(input('Enter the phrase to be neutralised, or Exit to quit\n'))