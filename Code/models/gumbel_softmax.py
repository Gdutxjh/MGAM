import torch
import torch.nn.functional as F

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape)
    U = U.to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    device = logits.device
    y = logits + sample_gumbel(logits.size(), device)
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=True):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard

# a = torch.tensor([[0.56,0.58,0.67],[0.1,0.9,0.2]]).cuda()
# a = torch.tensor([[0.56,0.58,0.67],[0.1,0.9,0.2]]).to("cuda:0")
# b = gumbel_softmax(a)
