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


def _check_no_grad(name, *args):
    """Raise if gradients are needed through a function computed by scipy.

    Scipy is given detached arrays, so that the gradients would be lost.
    """
    if _torch.is_grad_enabled() and any(
        _torch.is_tensor(arg) and arg.requires_grad for arg in args
    ):
        raise NotImplementedError(
            f"{name} cannot be differentiated with PyTorch: scipy computes it."
        )


class _Logm(_torch.autograd.Function):
    """Torch autograd function for matrix logarithm.

    Implementation based on:
    https://github.com/pytorch/pytorch/issues/9983#issuecomment-891777620
    """

    @staticmethod
    def _logm(x):
        mat_log = _gsnplinalg.logm(_to_numpy(x))
        return _torch.from_numpy(mat_log).to(x.device)

    @staticmethod
    def forward(ctx, tensor):
        """Apply matrix logarithm to a tensor."""
        ctx.save_for_backward(tensor)
        return _Logm._logm(tensor)

    @staticmethod
    def backward(ctx, grad):
        """Run gradients backward.

        The gradient is the upper right block of ``logm([[x^H, g], [0, x^H]])``.
        The block matrix is built with differentiable operations, and the
        logarithm goes through ``_Logm.apply``, so that the gradient can itself
        be differentiated.
        """
        (tensor,) = ctx.saved_tensors

        tensor_H = tensor.mT.conj().to(grad.dtype)
        n = tensor.size(-1)
        backward_tensor = _torch.cat(
            [
                _torch.cat([tensor_H, grad], dim=-1),
                _torch.cat([_torch.zeros_like(tensor_H), tensor_H], dim=-1),
            ],
            dim=-2,
        )

        return _Logm.apply(backward_tensor).to(tensor.dtype)[..., :n, n:]


logm = _cast_out_to_input_dtype(_Logm.apply)


def sqrtm(x):
    _check_no_grad("sqrtm", x)
    np_sqrtm = _scipy.linalg.sqrtm(_to_numpy(x))
    if np_sqrtm.dtype.kind == "c":
        np_sqrtm = np_sqrtm.astype(f"complex{int(np_sqrtm.dtype.name[7:]) // 2}")

    return _torch.from_numpy(np_sqrtm)


def svd(x, full_matrices=True, compute_uv=True):
    if compute_uv:
        return _torch.linalg.svd(x, full_matrices=full_matrices)

    return _torch.linalg.svdvals(x)


def vector_norm(x, axis=None, keepdims=False, ord=2):
    """Compute the vector norm.

    Parameters
    ----------
    x : array-like
        Array.
    axis : int or tuple[int] or None
        Axes over which to compute the norm. If None, all of them.
    keepdims : bool
        Whether to keep the reduced axes.
    ord : int or float
        Order of the norm.

    Returns
    -------
    norm : array-like
        Norm.
    """
    return _torch.linalg.vector_norm(x, ord=ord, dim=axis, keepdim=keepdims)


def matrix_norm(x, keepdims=False, ord="fro"):
    """Compute the matrix norm, over the last two axes.

    Parameters
    ----------
    x : array-like, shape=[..., n, m]
        Matrix.
    keepdims : bool
        Whether to keep the reduced axes.
    ord : str or int or float
        Order of the norm.

    Returns
    -------
    norm : array-like, shape=[...]
        Norm.
    """
    return _torch.linalg.matrix_norm(x, ord=ord, keepdim=keepdims)


def matrix_rank(a, tol=None, hermitian=False):
    """Compute the rank of matrices.

    Parameters
    ----------
    a : array-like, shape=[..., m, n]
        Matrix.
    tol : float
        Threshold below which singular values are considered zero. By
        default, the largest singular value times the largest dimension
        times the machine precision.
    hermitian : bool
        Whether the matrices are hermitian.

    Returns
    -------
    rank : array-like, shape=[...]
        Rank.
    """
    return _torch.linalg.matrix_rank(a, atol=tol, hermitian=hermitian)


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


def _solve_sylvester(a, b, q, tol):
    """Solve the Sylvester equation, without gradients."""
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
    return _torch.from_numpy(solution).to(q.device)


class _SolveSylvester(_torch.autograd.Function):
    """Torch autograd function for the solution of the Sylvester equation."""

    @staticmethod
    def forward(ctx, a, b, q, tol):
        """Solve the Sylvester equation."""
        x = _solve_sylvester(a, b, q, tol)
        ctx.save_for_backward(a, b, x)
        ctx.shapes = (a.shape, b.shape, q.shape)
        ctx.tol = tol
        return x

    @staticmethod
    def backward(ctx, grad):
        """Run gradients backward.

        If ``X`` solves ``A X + X B = Q`` and ``Y`` solves
        ``A^H Y + Y B^H = g``, the gradients with respect to ``A``, ``B`` and
        ``Q`` are ``-Y X^H``, ``-X^H Y`` and ``Y``. Unlike differentiating
        through the solver, this does not involve the eigenvectors of ``A``,
        whose gradient is undefined for repeated eigenvalues. ``Y`` goes
        through ``_SolveSylvester.apply``, so that the gradients can
        themselves be differentiated.
        """
        a, b, x = ctx.saved_tensors
        y = _SolveSylvester.apply(a.mT.conj(), b.mT.conj(), grad, ctx.tol)
        x_H = x.mT.conj()

        grads = (
            -y @ x_H if ctx.needs_input_grad[0] else None,
            -x_H @ y if ctx.needs_input_grad[1] else None,
            y if ctx.needs_input_grad[2] else None,
        )
        return tuple(
            None if grad_ is None else grad_.sum_to_size(shape)
            for grad_, shape in zip(grads, ctx.shapes)
        ) + (None,)


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
    return _SolveSylvester.apply(a, b, q, tol)


@_cast_out_to_input_dtype
def fractional_matrix_power(A, t):
    """Compute the fractional power of a matrix."""
    _check_no_grad("fractional_matrix_power", A, t)
    out = _scipy.linalg.fractional_matrix_power(_to_numpy(A), t)
    return _torch.from_numpy(out)


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
