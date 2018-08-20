"""Torch based random backend."""

import torch


def rand(*args, **kwargs):
    return torch.rand(*args, **kwargs)


def randint(*args, **kwargs):
    return torch.randint(*args, **kwargs)


def seed(*args, **kwargs):
    return torch.manual_seed(*args, **kwargs)


def normal(mean=0.0, std=1.0, shape=(1, 1)):
    return torch.normal(torch.ones(shape) * 0, torch.ones(shape) * 1)