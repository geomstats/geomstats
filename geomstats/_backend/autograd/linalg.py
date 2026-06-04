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
    matrix_power,
    matrix_rank,
    norm,
    svd,
)
from autograd.scipy.linalg import expm
from scipy.optimize import quadratic_assignment as _quadratic_assignment

from ._common import atol


def _adjoint(_ans, x, fn):
    vectorized = x.ndim == 3
    axes = (0, 2, 1) if vectorized else (1, 0)

    def vjp(g):
        n = x.shape[-1]
        size_m = x.shape[:-2] + (2 * n, 2 * n)
        mat = _np.zeros(size_m)
        mat[..., :n, :n] = x.transpose(axes)
        mat[..., n:, n:] = x.transpose(axes)
        mat[..., :n, n:] = g
        return fn(mat)[..., :n, n:]

    return vjp


_expm_vjp = _functools.partial(_adjoint, fn=expm)
_defvjp(expm, _expm_vjp)


def _transpose(array):
    axes = list(range(0, array.ndim))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    return _np.transpose(array, axes=axes)


def _is_symmetric(x, tol=atol):
    return (_np.abs(x - _transpose(x)) < tol).all()


def _is_hermitian(x, tol=atol):
    return (_np.abs(x - _np.conj(_transpose(x))) < tol).all()


_diag_vec = _np.vectorize(_np.diag, signature="(n)->(n,n)")

_logm_vec = _np.vectorize(_scipy.linalg.logm, signature="(n,m)->(n,m)")


def _logm(x):
    if _is_symmetric(x) and x.dtype not in [_np.complex64, _np.complex128]:
        eigvals, eigvecs = _np.linalg.eigh(x)
        if (eigvals > 0).all():
            eigvals = _np.log(eigvals)
            eigvals = _diag_vec(eigvals)
            transp_eigvecs = _transpose(eigvecs)
            result = _np.matmul(eigvecs, eigvals)
            result = _np.matmul(result, transp_eigvecs)
        else:
            result = _logm_vec(x)
    else:
        result = _logm_vec(x)

    return result


logm = _primitive(_logm)

_logm_vjp = _functools.partial(_adjoint, fn=logm)
_defvjp(logm, _logm_vjp)


def quadratic_assignment(a, b, options):
    return list(_quadratic_assignment(a, b, options=options).col_ind)


def solve_sylvester(a, b, q, tol=atol):
    if a.shape == b.shape:
        if _np.all(_np.isclose(a, b)) and _np.all(_np.abs(a - _transpose(a)) < tol):
            eigvals, eigvecs = _np.linalg.eigh(a)
            if _np.all(eigvals >= tol):
                tilde_q = _transpose(eigvecs) @ q @ eigvecs
                tilde_x = tilde_q / (eigvals[..., :, None] + eigvals[..., None, :])
                return eigvecs @ tilde_x @ _transpose(eigvecs)

    return _np.vectorize(
        _scipy.linalg.solve_sylvester, signature="(m,m),(n,n),(m,n)->(m,n)"
    )(a, b, q)


def sqrtm(x):
    return _np.vectorize(_scipy.linalg.sqrtm, signature="(n,m)->(n,m)")(x)


def qr(*args, **kwargs):
    return _np.vectorize(
        _np.linalg.qr, signature="(n,m)->(n,k),(k,m)", excluded=["mode"]
    )(*args, **kwargs)


def is_single_matrix_pd(mat):
    """Check if 2D square matrix is positive definite."""
    if mat.shape[0] != mat.shape[1]:
        return False
    if mat.dtype in [_np.complex64, _np.complex128]:
        if not _is_hermitian(mat):
            return False
        eigvals = _np.linalg.eigvalsh(mat)
        return _np.min(_np.real(eigvals)) > 0
    try:
        _np.linalg.cholesky(mat)
        return True
    except _np.linalg.LinAlgError as e:
        if e.args[0] == "Matrix is not positive definite":
            return False
        raise e


def fractional_matrix_power(A, t):
    if A.ndim == 2:
        return _scipy.linalg.fractional_matrix_power(A, t)

    return _np.stack([_scipy.linalg.fractional_matrix_power(A_, t) for A_ in A])


def polar(*args, **kwargs):
    """Polar decomposition of a matrix."""
    return _np.vectorize(
        _scipy.linalg.polar, signature="(n,n)->(n,n),(n,n)", excluded=["side"]
    )(*args, **kwargs)


def solve(a, b):
    """
    Solve a linear matrix equation, or system of linear scalar equations.

    Computes the "exact" solution, `x`, of the well-determined, i.e., full
    rank, linear matrix equation `ax = b`.

    Parameters
    ----------
    a : array-like, shape=[..., M, M]
        Coefficient matrix.
    b : array-like, shape=[..., M]
        Ordinate or "dependent variable" values".

    Returns
    -------
    x : array-like, shape=[..., M]
        Solution to the system a x = b.
    """
    batch_shape = a.shape[:-2]
    if batch_shape:
        b = _np.expand_dims(b, axis=-1)

    res = _np.linalg.solve(a, b)
    if batch_shape:
        return res[..., 0]

    return res
