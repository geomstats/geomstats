"""Pytorch based linear algebra backend."""

import torch
import numpy as np


def inv(*args, **kwargs):
    return torch.from_numpy(np.linalg.inv(*args, **kwargs))


def eigvalsh(*args, **kwargs):
    return torch.from_numpy(np.linalg.eigvalsh(*args, **kwargs))


def eigh(*args, **kwargs):
    eigs = np.linalg.eigh(*args, **kwargs)
    return torch.from_numpy(eigs[0]), torch.from_numpy(eigs[1])


def svd(*args, **kwargs):
    svds = np.linalg.svd(*args, **kwargs)
    return (torch.from_numpy(svds[0]),
            torch.from_numpy(svds[1]),
            torch.from_numpy(svds[2]))


def det(*args, **kwargs):
    return torch.from_numpy(np.linalg.det(*args, **kwargs))


def norm(x, ord=2, axis=None, keepdims=False):
    if axis is None:
        return torch.norm(x, p=ord)
    return torch.norm(x, p=ord, dim=axis)
