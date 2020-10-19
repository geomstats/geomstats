"""Pytorch based linear algebra backend."""

import numpy as np
import scipy.linalg
import torch


def _raise_not_implemented_error(*args, **kwargs):
    raise NotImplementedError


eig = _raise_not_implemented_error
logm = _raise_not_implemented_error
powerm = _raise_not_implemented_error


def sqrtm(x):
    np_sqrtm = np.vectorize(
        scipy.linalg.sqrtm, signature='(n,m)->(n,m)')(x)
    return torch.from_numpy(np_sqrtm)


def expm(x):
    np_expm = np.vectorize(
        scipy.linalg.expm, signature='(n,m)->(n,m)')(x)
    return torch.from_numpy(np_expm)


def inv(*args, **kwargs):
    return torch.from_numpy(np.linalg.inv(*args, **kwargs))


def eigvalsh(*args, **kwargs):
    return torch.from_numpy(np.linalg.eigvalsh(*args, **kwargs))


def eigh(*args, **kwargs):
    eigvals, eigvecs = torch.symeig(*args, eigenvectors=True, **kwargs)
    return eigvals, eigvecs


def svd(x, full_matrices=True, compute_uv=True):
    is_vectorized = x.ndim == 3
    axis = (0, 2, 1) if is_vectorized else (1, 0)
    if compute_uv:
        u, s, v_t = torch.svd(
            x, some=not full_matrices, compute_uv=compute_uv)
        return u, s, v_t.permute(axis)
    return torch.svd(x, some=not full_matrices, compute_uv=compute_uv)[1]


def det(*args, **kwargs):
    return torch.from_numpy(np.array(np.linalg.det(*args, **kwargs)))


def norm(x, ord=2, axis=None):
    if axis is None:
        return torch.norm(x, p=ord)
    return torch.norm(x, p=ord, dim=axis)


def qr(*args, **kwargs):
    matrix_q, matrix_r = np.vectorize(
        np.linalg.qr,
        signature='(n,m)->(n,k),(k,m)',
        excluded=['mode'])(*args, **kwargs)
    tensor_q = torch.from_numpy(matrix_q)
    tensor_r = torch.from_numpy(matrix_r)
    return tensor_q, tensor_r
