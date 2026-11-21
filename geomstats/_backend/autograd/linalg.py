"""Autograd based linear algebra backend."""

import autograd.numpy as _np
import autograd.scipy as _scipy
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

from ..numpy.linalg import logm as _logm
from ..numpy.linalg import quadratic_assignment  # NOQA
from ._common import atol

logm = _primitive(_logm)


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
