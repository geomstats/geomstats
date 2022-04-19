"""Pytorch based linear algebra backend."""

import numpy as np
import scipy.linalg
import scipy.optimize
import torch

from ..numpy import linalg as gsnplinalg


class Logm(torch.autograd.Function):
    """Torch autograd function for matrix logarithm.

    Implementation based on:
    https://github.com/pytorch/pytorch/issues/9983#issuecomment-891777620
    """

    @staticmethod
    def _logm(x):
        np_logm = gsnplinalg.logm(x.detach().cpu())
        torch_logm = torch.from_numpy(np_logm).to(x.device, dtype=x.dtype)
        return torch_logm

    @staticmethod
    def forward(ctx, tensor):
        """Apply matrix logarithm to a tensor."""
        ctx.save_for_backward(tensor)
        return Logm._logm(tensor)

    @staticmethod
    def backward(ctx, grad):
        """Run gradients backward."""
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


cholesky = torch.linalg.cholesky
eig = torch.linalg.eig
eigh = torch.linalg.eigh
eigvalsh = torch.linalg.eigvalsh
expm = torch.matrix_exp
inv = torch.inverse
det = torch.det
solve = torch.linalg.solve
qr = torch.linalg.qr
logm = Logm.apply


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


def quadratic_assignment(a, b, options):
    return list(scipy.optimize.quadratic_assignment(a, b, options=options).col_ind)


def solve_sylvester(a, b, q):
    if a.shape == b.shape:
        if torch.all(a == b) and torch.all(torch.abs(a - a.transpose(-2, -1)) < 1e-6):
            eigvals, eigvecs = eigh(a)
            if torch.all(eigvals >= 1e-6):
                tilde_q = eigvecs.transpose(-2, -1) @ q @ eigvecs
                tilde_x = tilde_q / (eigvals[..., :, None] + eigvals[..., None, :])
                return eigvecs @ tilde_x @ eigvecs.transpose(-2, -1)

            conditions = torch.all(eigvals >= 1e-6) or (
                a.shape[-1] >= 2.0
                and torch.all(eigvals[..., 0] > -1e-6)
                and torch.all(eigvals[..., 1] >= 1e-6)
                and torch.all(torch.abs(q + q.transpose(-2, -1)) < 1e-6)
            )
            if conditions:
                tilde_q = eigvecs.transpose(-2, -1) @ q @ eigvecs
                tilde_x = tilde_q / (
                    eigvals[..., :, None]
                    + eigvals[..., None, :]
                    + torch.eye(a.shape[-1])
                )
                return eigvecs @ tilde_x @ eigvecs.transpose(-2, -1)

    solution = np.vectorize(
        scipy.linalg.solve_sylvester, signature="(m,m),(n,n),(m,n)->(m,n)"
    )(a, b, q)
    return torch.from_numpy(solution)


# (TODO) (sait) torch.linalg.cholesky_ex for even faster way
def is_single_matrix_pd(mat):
    """Check if 2D square matrix is positive definite."""
    if mat.shape[0] != mat.shape[1]:
        return False
    try:
        torch.linalg.cholesky(mat)
        return True
    except RuntimeError:
        return False
