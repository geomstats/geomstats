"""Torch based random backend."""

import torch


def rand(*args, **kwargs):
    return torch.rand(*args, **kwargs)


def randint(*args, **kwargs):
    return torch.randint(*args, **kwargs)


def seed(*args, **kwargs):
    return torch.manual_seed(*args, **kwargs)


def normal(loc=0.0, scale=1.0, size=(1,)):
    if not hasattr(size, '__iter__'):
        size = (size,)
    return torch.normal(mean=loc, std=scale, size=size)


def uniform(low=0.0, high=1.0, size=None):
    if size is None:
        size = (1,)
    if not hasattr(size, '__iter__'):
        size = (size,)
    assert low <= high
    return (high - low) * torch.rand(*size) + low
