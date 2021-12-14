"""Pytorch based linear algebra backend."""

import numpy as np
import scipy.linalg
import torch

from ..numpy import linalg as gsnplinalg


class Logm(torch.autograd.Function):
    """
    Torch autograd function for matrix logarithm.
    Implementation based on:
    https://github.com/pytorch/pytorch/issues/9983#issuecomment-891777620
    """

    @staticmethod
    def _logm(x):
        np_logm = gsnplinalg.logm(x.detach().cpu())
        torch_logm = torch.from_numpy(np_logm).to(x.device)
        return torch_logm

    @staticmethod
    def forward(ctx, tensor):
        ctx.save_for_backward(tensor)
        return Logm._logm(tensor)

    @staticmethod
    def backward(ctx, grad):
        (tensor,) = ctx.saved_tensors

        vectorized = tensor.ndim == 3
        axes = (0, 2, 1) if vectorized else (1, 0)
        tensor_H = tensor.permute(axes).conj().to(grad.dtype)
        n = tensor.size(-1)
        bshape = tensor.shape[:-2] + (2 * n, 2 * n)
        backward_tensor = torch.zeros(*bshape, dtype=grad.dtype, device=grad.device)
        backward_tensor[..., :n, :n] = tensor_H
        backward_tensor[..., n:, n:] = tensor_H
        backward_tensor[..., :n, n:] = grad

        return Logm._logm(backward_tensor).to(tensor.dtype)[..., :n, n:]


eig = torch.linalg.eig
eigh = torch.linalg.eigh
eigvalsh = torch.linalg.eigvalsh
expm = torch.matrix_exp
inv = torch.inverse
det = torch.det
solve = torch.linalg.solve
qr = torch.linalg.qr
logm = Logm.apply


def cholesky(a):
    return torch.cholesky(a, upper=False)


def sqrtm(x):
    np_sqrtm = np.vectorize(scipy.linalg.sqrtm, signature="(n,m)->(n,m)")(x)
    return torch.as_tensor(np_sqrtm, dtype=x.dtype)


def svd(x, full_matrices=True, compute_uv=True):
    is_vectorized = x.ndim == 3
    axis = (0, 2, 1) if is_vectorized else (1, 0)
    if compute_uv:
        u, s, v_t = torch.svd(x, some=not full_matrices, compute_uv=compute_uv)
        return u, s, v_t.permute(axis)
    return torch.svd(x, some=not full_matrices, compute_uv=compute_uv)[1]


def norm(x, ord=None, axis=None):
    if axis is None:
        return torch.linalg.norm(x, ord=ord)
    return torch.linalg.norm(x, ord=ord, dim=axis)


def matrix_rank(a, hermitian=False, **_unused_kwargs):
    return torch.linalg.matrix_rank(a, hermitian)


def solve_sylvester(a, b, q):
    if a.shape == b.shape:
        if torch.all(a == b) and torch.all(torch.abs(a - a.transpose(-2, -1)) < 1e-6):
            eigvals, eigvecs = eigh(a)
            if torch.all(eigvals >= 1e-6):
                tilde_q = eigvecs.transpose(-2, -1) @ q @ eigvecs
                tilde_x = tilde_q / (eigvals[..., :, None] + eigvals[..., None, :])
                return eigvecs @ tilde_x @ eigvecs.transpose(-2, -1)

    solution = np.vectorize(
        scipy.linalg.solve_sylvester, signature="(m,m),(n,n),(m,n)->(m,n)"
    )(a, b, q)
    return torch.from_numpy(solution)
