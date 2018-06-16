"""Pytorch based linear algebra backend."""

import torch


def eigvalsh(*args, **kwargs):
    return torch.eig(*args, **kwargs)[0][:, 0]

def svd(*args, **kwargs):
    return torch.svd(*args, **kwargs)

def det(*args, **kwargs):
    return torch.det(*args, **kwargs)

def norm(x, ord=2, axis=None, keepdims=False):
    assert keepdims is False
    if axis is None:
        return torch.norm(x, p=ord)
    return torch.norm(x, p=ord, dim=axis)
