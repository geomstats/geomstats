"""Autograd based linear algebra backend."""

import functools as _functools

import autograd.numpy as _np
import autograd.scipy as _scipy
from autograd.extend import defvjp as _defvjp
from autograd.extend import primitive as _primitive
from autograd.numpy.linalg import (
    cholesky,
    det,
    eig,
    eigh,
    eigvalsh,
    inv,
    matrix_norm,
    matrix_power,
    matrix_rank,
    qr,
    svd,
    vector_norm,
)
from autograd.scipy.linalg import expm, fractional_matrix_power, polar, sqrtm

from ..numpy.linalg import quadratic_assignment  # NOQA
from ._common import atol


def _adjoint(_ans, x, fn):
    """Get the VJP of a matrix function.

    The VJP is the upper right block of ``fn([[x^T, g], [0, x^T]])``. The block
    matrix is built with differentiable operations, so that the VJP can itself
    be differentiated.
    """
    transposed_x = _np.matrix_transpose(x)
    zeros = _np.zeros_like(transposed_x)
    n = x.shape[-1]

    def vjp(g):
        mat = _np.concatenate(
            [
                _np.concatenate([transposed_x, g], axis=-1),
                _np.concatenate([zeros, transposed_x], axis=-1),
            ],
            axis=-2,
        )
        return fn(mat)[..., :n, n:]

    return vjp


_expm_vjp = _functools.partial(_adjoint, fn=expm)
_defvjp(expm, _expm_vjp)


def _logm(x):
    """Compute the matrix logarithm.

    Real symmetric positive definite matrices use their eigendecomposition.
    The other matrices use scipy, matrix by matrix, which may give a complex
    logarithm.

    Parameters
    ----------
    x : array-like, shape=[..., n, n]
        Matrix.

    Returns
    -------
    logm : array-like, shape=[..., n, n]
        Matrix logarithm.
    """
    if _np.iscomplexobj(x):
        return _scipy.linalg.logm(x)

    n = x.shape[-1]
    mats = _np.reshape(x, (-1, n, n))

    is_sym = _np.all(_np.abs(mats - _np.matrix_transpose(mats)) < atol, axis=(-2, -1))
    is_spd = _np.zeros_like(is_sym)
    if _np.any(is_sym):
        eigvals, eigvecs = _np.linalg.eigh(mats[is_sym])
        is_pos = _np.all(eigvals > 0, axis=-1)
        is_spd[is_sym] = is_pos

        eigvals, eigvecs = eigvals[is_pos], eigvecs[is_pos]
        spd_logm = _np.matmul(
            eigvecs * _np.log(eigvals)[..., None, :], _np.matrix_transpose(eigvecs)
        )

    if _np.all(is_spd):
        return _np.reshape(spd_logm, x.shape)

    other_logm = _scipy.linalg.logm(mats[~is_spd])
    result = _np.empty(mats.shape, dtype=_np.result_type(mats, other_logm))
    if _np.any(is_spd):
        result[is_spd] = spd_logm
    result[~is_spd] = other_logm
    return _np.reshape(result, x.shape)


logm = _primitive(_logm)

_logm_vjp = _functools.partial(_adjoint, fn=logm)
_defvjp(logm, _logm_vjp)


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
        and _np.allclose(a, b, atol=tol)
        and _np.all(_np.abs(a - _np.matrix_transpose(a)) < tol)
    ):
        eigvals, eigvecs = _np.linalg.eigh(a)
        if _np.all(eigvals > -tol):
            tilde_q = _np.matrix_transpose(eigvecs) @ q @ eigvecs
            denom = eigvals[..., :, None] + eigvals[..., None, :]
            is_zero = _np.abs(denom) <= tol
            tilde_x = _np.where(
                is_zero, 0.0, tilde_q / _np.where(is_zero, 1.0, denom)
            )
            return eigvecs @ tilde_x @ _np.matrix_transpose(eigvecs)

    return _scipy.linalg.solve_sylvester(a, b, q)


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
    return _np.linalg.solve(a, _np.expand_dims(b, axis=-1))[..., 0]
