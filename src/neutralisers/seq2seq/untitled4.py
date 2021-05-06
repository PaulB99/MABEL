import model
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
m = model.seq2seq(device, 30522)

pp = 0
for p in list(m.parameters()):
    nn=1
    for s in list(p.size()):
        nn = nn*s
    pp += nn
print(pp)