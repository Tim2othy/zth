import torch
import torch.nn.functional as F


def train(train_range, parameters, Xtr, Ytr, itos):
    print("training net")
    C, W1, b1, W2, b2 = parameters
    lossi = []
    stepi = []

    for i in range(train_range):

        # minibatch
        xi = torch.randint(0, Xtr.shape[0], (96,))

        # forward pass
        emb = C[Xtr[xi]]  # (32, 3, 10)
        h = torch.tanh(emb.view(-1, 30) @ W1 + b1)  # (32, 200)
        logits = h @ W2 + b2  # (32, 27)
        loss = F.cross_entropy(logits, Ytr[xi])

        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # update
        # lr = lrs[i]
        lr = 0.1 if i < 12000 else 0.01
        for p in parameters:
            if p.grad is not None:
                p.data += -lr * p.grad

        # track stats
        stepi.append(i)
        lossi.append(loss.log10().item())

    # Save plot of the loss

    # plot = plt.plot(stepi, lossi)
    # plt.savefig(plot, "loss_plot.png")


def eval(Xdev, Ydev, parameters):
    print("evaluating net")
    C, W1, b1, W2, b2 = parameters
    emb = C[Xdev]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ydev)
    print(loss)

    # This is the loss for the latest minibatch
    if loss is not None:
        print(loss.item())


def sample(parameters, itos, block_size):
    print("sampling from net")
    C, W1, b1, W2, b2 = parameters

    for _ in range(20):

        out = []
        context = [0] * block_size  # initialize with all ...
        while True:
            emb = C[torch.tensor([context])]  # (1,block_size,d)
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break

        print("".join(itos[i] for i in out))
