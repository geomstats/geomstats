"""Pytorch based linear algebra backend."""

import scipy as _scipy
import torch as _torch
from torch.linalg import (
    cholesky,
    det,
    eig,
    eigh,
    eigvalsh,
    inv,
    matrix_power,
    qr,
)
from torch.linalg import matrix_exp as expm

from ..numpy import linalg as _gsnplinalg
from ._common import atol
from ._dtype import _cast_out_to_input_dtype


def _to_numpy(x):
    """Convert a tensor to a numpy array, for scipy.

    Scipy is not given tensors: with ``SCIPY_ARRAY_API=1``, it would run its
    numpy-based code on them.
    """
    return x.detach().cpu().numpy() if _torch.is_tensor(x) else x


class _Logm(_torch.autograd.Function):
    """Torch autograd function for matrix logarithm.

    Implementation based on:
    https://github.com/pytorch/pytorch/issues/9983#issuecomment-891777620
    """

    @staticmethod
    def _logm(x):
        mat_log = _gsnplinalg.logm(x.detach().cpu().numpy())
        return _torch.from_numpy(mat_log).to(x.device)

    @staticmethod
    def forward(ctx, tensor):
        """Apply matrix logarithm to a tensor."""
        ctx.save_for_backward(tensor)
        return _Logm._logm(tensor)

    @staticmethod
    def backward(ctx, grad):
        """Run gradients backward."""
        (tensor,) = ctx.saved_tensors

        tensor_H = tensor.mT.conj().to(grad.dtype)
        n = tensor.size(-1)
        bshape = tensor.shape[:-2] + (2 * n, 2 * n)
        backward_tensor = _torch.zeros(*bshape, dtype=grad.dtype, device=grad.device)
        backward_tensor[..., :n, :n] = tensor_H
        backward_tensor[..., n:, n:] = tensor_H
        backward_tensor[..., :n, n:] = grad

        return _Logm._logm(backward_tensor).to(tensor.dtype)[..., :n, n:]


logm = _cast_out_to_input_dtype(_Logm.apply)


def sqrtm(x):
    np_sqrtm = _scipy.linalg.sqrtm(_to_numpy(x))
    if np_sqrtm.dtype.kind == "c":
        np_sqrtm = np_sqrtm.astype(f"complex{int(np_sqrtm.dtype.name[7:]) // 2}")

    return _torch.from_numpy(np_sqrtm)


def svd(x, full_matrices=True, compute_uv=True):
    if compute_uv:
        return _torch.linalg.svd(x, full_matrices=full_matrices)

    return _torch.linalg.svdvals(x)


def norm(x, ord=None, axis=None, keepdims=False):
    return _torch.linalg.norm(x, ord=ord, dim=axis, keepdim=keepdims)


def matrix_rank(a, hermitian=False, **_unused_kwargs):
    return _torch.linalg.matrix_rank(a, hermitian=hermitian)


def quadratic_assignment(a, b, options=None):
    """Solve the quadratic assignment problem.

    Parameters
    ----------
    a : array-like, shape=[..., n, n]
        First matrix.
    b : array-like, shape=[..., n, n]
        Second matrix.
    options : dict
        Options of ``scipy.optimize.quadratic_assignment``.

    Returns
    -------
    perm : array-like, shape=[..., n]
        Permutation of the nodes of ``b`` that is optimal for ``a``.
    """
    perm = _gsnplinalg.quadratic_assignment(
        _to_numpy(a), _to_numpy(b), options=options
    )
    return _torch.from_numpy(perm)


def solve_sylvester(a, b, q, tol=atol):
    """Solve the Sylvester equation ``A X + X B = Q``.

    If ``A`` and ``B`` are the same symmetric positive semidefinite matrix,
    the equation is solved in the eigenbasis of ``A``. The entries whose
    eigenvalue sum vanishes are set to zero, which gives the minimum-norm
    solution. Otherwise, scipy is used, matrix by matrix.

    Parameters
    ----------
    a : array-like, shape=[..., m, m]
        First coefficient matrix, ``A``.
    b : array-like, shape=[..., n, n]
        Second coefficient matrix, ``B``.
    q : array-like, shape=[..., m, n]
        Right-hand side, ``Q``.
    tol : float
        Tolerance on equality, symmetry and eigenvalues.

    Returns
    -------
    x : array-like, shape=[..., m, n]
        Solution, ``X``.
    """
    if (
        a.shape == b.shape
        and _torch.all(_torch.abs(a - b) < tol)
        and _torch.all(_torch.abs(a - a.mT) < tol)
    ):
        eigvals, eigvecs = eigh(a)
        if _torch.all(eigvals > -tol):
            tilde_q = eigvecs.mT @ q @ eigvecs
            denom = eigvals[..., :, None] + eigvals[..., None, :]
            is_zero = _torch.abs(denom) <= tol
            tilde_x = _torch.where(
                is_zero,
                _torch.zeros_like(tilde_q),
                tilde_q / _torch.where(is_zero, _torch.ones_like(denom), denom),
            )
            return eigvecs @ tilde_x @ eigvecs.mT

    solution = _scipy.linalg.solve_sylvester(_to_numpy(a), _to_numpy(b), _to_numpy(q))
    return _torch.from_numpy(solution)


@_cast_out_to_input_dtype
def fractional_matrix_power(A, t):
    """Compute the fractional power of a matrix."""
    out = _scipy.linalg.fractional_matrix_power(_to_numpy(A), t)
    return _torch.from_numpy(out)


def polar(*args, **kwargs):
    """Polar decomposition of a matrix."""
    args = [_to_numpy(arg) for arg in args]
    unitary, hermitian = _scipy.linalg.polar(*args, **kwargs)
    return _torch.from_numpy(unitary), _torch.from_numpy(hermitian)


def solve(a, b):
    """Solve a linear system with vector right-hand sides.

    Parameters
    ----------
    a : array-like, shape=[..., M, M]
        Full-rank coefficient matrix.
    b : array-like, shape=[..., M]
        Right-hand-side vector. Batch dimensions must broadcast with `a`.

    Returns
    -------
    x : array-like, shape=[..., M]
        Solution to a x = b, with broadcast batch dimensions.
    """
    return _torch.linalg.solve(a, b.unsqueeze(-1)).squeeze(-1)
