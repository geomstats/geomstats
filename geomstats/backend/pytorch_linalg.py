"""Pytorch based linear algebra backend."""

import torch
import numpy as np


def eigvalsh(*args, **kwargs):
    return torch.from_numpy(np.linalg.eigvalsh(*args, **kwargs))


def eigh(*args, **kwargs):
    eigs = np.linalg.eigh(*args, **kwargs)
    return torch.from_numpy(eigs[0]), torch.from_numpy(eigs[1])


def svd(*args, **kwargs):
    return torch.svd(*args, **kwargs)


def det(*args, **kwargs):
    return torch.det(*args, **kwargs)


def norm(x, ord=2, axis=None, keepdims=False):
    assert keepdims is False
    if axis is None:
        return torch.norm(x, p=ord)
    return torch.norm(x, p=ord, dim=axis)
