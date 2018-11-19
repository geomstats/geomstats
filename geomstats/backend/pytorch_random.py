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
