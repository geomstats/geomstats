"""Torch based random backend."""

import torch
from torch import distributions

def rand(*args, **kwargs):
    return torch.rand(*args, **kwargs)


def randint(*args, **kwargs):
    return torch.randint(*args, **kwargs)


def seed(*args, **kwargs):
    return torch.manual_seed(*args, **kwargs)


def normal(loc=0.0, scale=1.0, size=(1, 1)):
    return torch.normal(torch.zeros(size), torch.ones(size))


def choice(a, size=None, replace=True, p=None):
    if type(a) == int:
        a = torch.arange(a)
    if type(size) == int:
        size_prod = size
    elif(size is None):
        size_prod = 1
    else:
        size_prod = torch.Tensor(list(size)).prod()

    if(p is None):
        p = torch.ones(len(a))/len(a)
    sample = a[torch.multinomial(p, size_prod, replacement=replace)]
    return sample.reshape(size)
