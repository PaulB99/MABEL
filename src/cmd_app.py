import joint.run as run
import sys
# Run the command line app

if __name__ == "__main__":
    args = sys.argv
    acceptable_taggers = ['base_model','large_model', 'lexi']
    acceptable_neutralisers = ['bart','roberta','parrot']
    if not len(args) == 2:
        raise ValueError('Arguments must be 2 valid models')
        
    tagger = str(args[0])
    neu = str(args[1])
        
    r = run.runner(tagger, neu)
    sentence = str(input('Enter the phrase to be neutralised, or Exit to quit\n'))
    while not sentence == 'Exit':
        print(r.pipeline(sentence))
        sentence = str(input('Enter the phrase to be neutralised, or Exit to quit\n'))