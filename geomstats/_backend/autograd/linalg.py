"""Autograd based linear algebra backend."""

import functools as _functools

import autograd.numpy as _np
import autograd.scipy.linalg as _asp
import scipy as _scipy
from autograd.extend import defvjp as _defvjp
from autograd.extend import primitive as _primitive
from autograd.numpy.linalg import (  # NOQA
    cholesky,
    det,
    eig,
    eigh,
    eigvalsh,
    inv,
    matrix_rank,
    norm,
    solve,
    svd,
)

from ._common import to_ndarray as _to_ndarray
from ._dtype_wrapper import _cast_fout_from_input_dtype

_diag_vec = _np.vectorize(_np.diag, signature="(n)->(n,n)")

_logm_vec = _cast_fout_from_input_dtype(
    target=_np.vectorize(_scipy.linalg.logm, signature="(n,m)->(n,m)")
)


def _is_symmetric(x, tol=1e-12):
    new_x = _to_ndarray(x, to_ndim=3)
    return (_np.abs(new_x - _np.transpose(new_x, axes=(0, 2, 1))) < tol).all()


@_primitive
def expm(x):
    return _np.vectorize(_asp.expm, signature="(n,m)->(n,m)")(x)


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


@_primitive
def logm(x):
    ndim = x.ndim
    new_x = _to_ndarray(x, to_ndim=3)
    if _is_symmetric(new_x):
        eigvals, eigvecs = _np.linalg.eigh(new_x)
        if (eigvals > 0).all():
            eigvals = _np.log(eigvals)
            eigvals = _diag_vec(eigvals)
            transp_eigvecs = _np.transpose(eigvecs, axes=(0, 2, 1))
            result = _np.matmul(eigvecs, eigvals)
            result = _np.matmul(result, transp_eigvecs)
        else:
            result = _logm_vec(new_x)
    else:
        result = _logm_vec(new_x)

    if ndim == 2:
        return result[0]
    return result


_logm_vjp = _functools.partial(_adjoint, fn=logm)
_defvjp(logm, _logm_vjp)


def solve_sylvester(a, b, q):
    if a.shape == b.shape:
        axes = (0, 2, 1) if a.ndim == 3 else (1, 0)
        if _np.all(a == b) and _np.all(_np.abs(a - _np.transpose(a, axes)) < 1e-12):
            eigvals, eigvecs = eigh(a)
            if _np.all(eigvals >= 1e-12):
                tilde_q = _np.transpose(eigvecs, axes) @ q @ eigvecs
                tilde_x = tilde_q / (eigvals[..., :, None] + eigvals[..., None, :])
                return eigvecs @ tilde_x @ _np.transpose(eigvecs, axes)

    return _np.vectorize(
        _scipy.linalg.solve_sylvester, signature="(m,m),(n,n),(m,n)->(m,n)"
    )(a, b, q)


@_cast_fout_from_input_dtype
def sqrtm(x):
    return _np.vectorize(_scipy.linalg.sqrtm, signature="(n,m)->(n,m)")(x)


def quadratic_assignment(a, b, options):
    return list(_scipy.optimize.quadratic_assignment(a, b, options=options).col_ind)


def qr(*args, **kwargs):
    return _np.vectorize(
        _np.linalg.qr, signature="(n,m)->(n,k),(k,m)", excluded=["mode"]
    )(*args, **kwargs)


def is_single_matrix_pd(mat):
    """Check if a 2D square matrix is positive definite."""
    if mat.shape[0] != mat.shape[1]:
        return False
    try:
        _np.linalg.cholesky(mat)
        return True
    except _np.linalg.LinAlgError as e:
        if e.args[0] == "Matrix is not positive definite":
            return False
        raise e
