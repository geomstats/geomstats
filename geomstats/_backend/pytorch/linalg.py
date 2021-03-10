"""Pytorch based linear algebra backend."""

import numpy as np
import scipy.linalg
import torch


def _raise_not_implemented_error(*args, **kwargs):
    raise NotImplementedError


eig = _raise_not_implemented_error
expm = torch.matrix_exp
logm = _raise_not_implemented_error
inv = torch.inverse
det = torch.det


def cholesky(a):
    return torch.cholesky(a, upper=False)


def sqrtm(x):
    np_sqrtm = np.vectorize(
        scipy.linalg.sqrtm, signature='(n,m)->(n,m)')(x)
    return torch.as_tensor(np_sqrtm, dtype=x.dtype)


def eigvalsh(a, **kwargs):
    upper = False
    if 'UPLO' in kwargs:
        upper = (kwargs['UPLO'] == 'U')
    return torch.symeig(a, eigenvectors=False, upper=upper)[0]


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


def norm(x, ord=None, axis=None):
    if axis is None:
        return torch.linalg.norm(x, ord=ord)
    return torch.linalg.norm(x, ord=ord, dim=axis)


def solve_sylvester(a, b, q):
    if a.shape == b.shape:
        if torch.all(a == b) and torch.all(
                torch.abs(a - a.transpose(-2, -1)) < 1e-6):
            eigvals, eigvecs = eigh(a)
            if torch.all(eigvals >= 1e-6):
                tilde_q = eigvecs.transpose(-2, -1) @ q @ eigvecs
                tilde_x = tilde_q / (
                    eigvals[..., :, None] + eigvals[..., None, :])
                return eigvecs @ tilde_x @ eigvecs.transpose(-2, -1)

    solution = np.vectorize(
        scipy.linalg.solve_sylvester,
        signature='(m,m),(n,n),(m,n)->(m,n)')(a, b, q)
    return torch.from_numpy(solution)


def qr(*args, **kwargs):
    matrix_q, matrix_r = np.vectorize(
        np.linalg.qr,
        signature='(n,m)->(n,k),(k,m)',
        excluded=['mode'])(*args, **kwargs)
    tensor_q = torch.from_numpy(matrix_q)
    tensor_r = torch.from_numpy(matrix_r)
    return tensor_q, tensor_r
