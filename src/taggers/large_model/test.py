# Test the basic model
import torch
import model
from torchtext.data import Field, TabularDataset, BucketIterator
from transformers import BertTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import csv
import os
import re
folder = re.split("\\\\|/",os.getcwd())[-5]
if folder=='pab734':
    os.environ['TORCH_HOME'] = '/data/labfs/pab734/torch'

# Probably wants to go into a nice util file
def load_ckpt(load_path, model):
    #state_dict = torch.load(load_path, map_location=device)
    #model.load_state_dict(state_dict['model_state_dict'])
    model.load_state_dict(torch.load(load_path))
    print(f'Trained model loaded from <== {load_path}')
    #return state_dict['valid_loss']

# Test given model using dataset
def test(model, test_loader):
    y_pred = []
    y_target = []
    # Store each classification
    bias_corr = []
    bias_inco = []
    un_corr = []
    un_inco = []
    model.eval()
    with torch.no_grad():
        for (labels, text), _ in test_loader:
            labels = labels.type(torch.LongTensor)           
            labels = labels.to(device)
            text = text.type(torch.LongTensor)  
            text = text.to(device)
            output = model(labels, text)
            _, output = output
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_target.extend(labels.tolist())
            if torch.argmax(output, 1) == labels:
                if labels == 1:
                    bias_corr.append(text)
                else:
                    un_corr.append(text)
            else:
                if labels == 1:
                    un_inco.append(text)
                else:
                    bias_inco.append(text)
    
    # Print report
    print('Classification Report:\n')
    print(classification_report(y_target, y_pred, labels=[1,0], digits=4))
    
    # Print examples
    print('Correctly identified as biased:')
    print(bias_corr[:10])
    
    print('Correctly identified as unbiased:')
    print(un_corr[:10])
    
    print('Incorrectly identified as biased:')
    print(bias_inco[:10])
    
    print('Inorrectly identified as unbiased:')
    print(un_inco[:10])
    
    with open('examples.csv', 'w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Correct biased", "Correct unbiased", "Incorrect biased", "Incorrect unbiased"])
        for n in range(200):
            writer.writerow([bias_corr[n], un_corr[n], bias_inco[n], un_inco[n]])
            
    # Create confusion matrix
    cm = confusion_matrix(y_target, y_pred, labels=[1,0])
    fig, ax = plt.subplots()

    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Target')

    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])
    
    plt.savefig('confusion.png')
    
# Run the test
if __name__ == "__main__":
    
    data_path = '../../../data/'
    
    # Tokeniser
    tokeniser = BertTokenizer.from_pretrained('bert-large-uncased')
    
    # Check if GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    MAX_SEQ_LEN = 128
    PAD_INDEX = tokeniser.convert_tokens_to_ids(tokeniser.pad_token)
    UNK_INDEX = tokeniser.convert_tokens_to_ids(tokeniser.unk_token)
    
    # Fields
    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(use_vocab=False, tokenize=tokeniser.encode, lower=False, include_lengths=False, batch_first=True, fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
    fields = [('label', label_field), ('text', text_field)]
    
    # Load in data 
    test_data, train = TabularDataset.splits(path=data_path, test='datasets/main/test_detection.csv', train='datasets/main/train_detection.csv',format='CSV', fields=fields, skip_header=True)
    
    # Test data iterator
    test_iter = BucketIterator(test_data, batch_size=16, device=device, train=False, shuffle=False, sort=False, sort_key=lambda x: len(x.text))
    
    # Test model
    mymodel = model.BERT().to(device)
    load_ckpt('../../../cache/taggers/large_model.pt', mymodel)
    test(mymodel, test_iter)
    
