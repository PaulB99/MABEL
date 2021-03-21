import torch 
import _ as t_model
import _ as n_model


# Program to load pretrained models and run the full pipeline


def load_ckpt(load_path, model):
    model.load_state_dict(torch.load(load_path))
    print(f'Trained model loaded from <== {load_path}')

def pipeline(sentence, tagger, neutraliser):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    tagger_path = '../../cache/taggers/' + tagger + '.pt'
    tagger_model = t_model.BERT().to(device)
    load_ckpt(tagger_path, tagger_model)
    print('Tagger loaded!')
    
    neutraliser_path = '../../cache/neutralisers/' + neutraliser + '.pt'
    neutraliser_model = n_model.BART().to(device)
    load_ckpt(neutraliser_path, neutraliser_model)
    print('Neutraliser loaded!')
    
    
    