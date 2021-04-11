import joint.run as run
import sys
# Run the command line app

if __name__ == "__main__":
    args = str(sys.argv)
    r = run.runner('base_model', 'bart')
    sentence = str(input('Enter the phrase to be neutralised, or Exit to quit\n'))
    while not sentence == 'Exit':
        print(r.pipeline(sentence))
        sentence = str(input('Enter the phrase to be neutralised, or Exit to quit\n'))