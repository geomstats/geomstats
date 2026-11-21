"""Numpy based linear algebra backend."""

import numpy as _np
import scipy as _scipy
from numpy.linalg import (
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
from scipy.linalg import expm, fractional_matrix_power, polar, sqrtm

from ._common import atol


def logm(x):
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
    batch_shape = _np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    if not batch_shape:
        return _scipy.optimize.quadratic_assignment(a, b, options=options).col_ind

    n = a.shape[-1]
    a = _np.reshape(_np.broadcast_to(a, batch_shape + (n, n)), (-1, n, n))
    b = _np.reshape(_np.broadcast_to(b, batch_shape + (n, n)), (-1, n, n))
    perms = [
        _scipy.optimize.quadratic_assignment(a_, b_, options=options).col_ind
        for a_, b_ in zip(a, b)
    ]
    return _np.reshape(_np.stack(perms), batch_shape + (n,))


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
