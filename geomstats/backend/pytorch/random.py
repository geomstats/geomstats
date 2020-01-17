"""Torch based random backend."""

import torch


def rand(*args, **kwargs):
    return torch.rand(*args, **kwargs)


def randint(*args, **kwargs):
    return torch.randint(*args, **kwargs)


def seed(*args, **kwargs):
    return torch.manual_seed(*args, **kwargs)


def normal(loc=0.0, scale=1.0, size=(1, 1)):
    return torch.normal(torch.zeros(size), torch.ones(size))


def choice(a, size=None, replace=True, p=None):
    if isinstance(a, int):
        a = torch.arange(a).float()

    size_prod = size
    if size is None:
        size_prod = 1

    elif not isinstance(size, int):
        size_prod = int(torch.prod(torch.Tensor(size)).item())

    if p is None:
        p = torch.ones_like(a) / float(len(a))
    sample = a[torch.multinomial(p, size_prod, replacement=replace)]
    return sample.reshape(size)
