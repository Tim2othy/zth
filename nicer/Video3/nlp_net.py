import torch
from data import build_dataset, words, build_vocab
from functions import eval, sample, train

# context length: how many characters do we take to predict the next one?
block_size = 3
train_range = 15000

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

stoi, itos = build_vocab(words)

Xtr, Ytr = build_dataset(words[:n1], stoi, block_size)
Xdev, Ydev = build_dataset(words[n1:n2], stoi, block_size)
Xte, Yte = build_dataset(words[n2:], stoi, block_size)

C = torch.randn((27, 10))
W1 = torch.randn((30, 180))
b1 = torch.randn(180)
W2 = torch.randn((180, 27))
b2 = torch.randn(27)
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True


train(train_range, parameters, Xtr, Ytr, itos)
eval(Xdev, Ydev, parameters)
sample(parameters, itos, block_size)
