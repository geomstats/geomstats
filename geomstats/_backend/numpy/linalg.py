"""Numpy based linear algebra backend."""

import functools

import numpy as _np
import scipy as _scipy
from numpy.linalg import (  # NOQA
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

from ._common import atol, cast
from ._common import to_ndarray as _to_ndarray
from ._dtype_wrapper import _cast_fout_from_input_dtype

_diag_vec = _np.vectorize(_np.diag, signature="(n)->(n,n)")

_logm_vec = _cast_fout_from_input_dtype(
    _np.vectorize(_scipy.linalg.logm, signature="(n,m)->(n,m)")
)


def _is_symmetric(x, tol=atol):
    new_x = _to_ndarray(x, to_ndim=3)
    return (_np.abs(new_x - _np.transpose(new_x, axes=(0, 2, 1))) < tol).all()


def expm(x):
    return _np.vectorize(_scipy.linalg.expm, signature="(n,m)->(n,m)")(x)


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


def solve_sylvester(a, b, q, tol=atol):
    if a.shape == b.shape:
        axes = (0, 2, 1) if a.ndim == 3 else (1, 0)
        if _np.all(a == b) and _np.all(_np.abs(a - _np.transpose(a, axes)) < tol):
            eigvals, eigvecs = eigh(a)
            if _np.all(eigvals >= tol):
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
    """Check if 2D square matrix is positive definite."""
    if mat.shape[0] != mat.shape[1]:
        return False
    try:
        _np.linalg.cholesky(mat)
        return True
    except _np.linalg.LinAlgError as e:
        if e.args[0] == "Matrix is not positive definite":
            return False
        raise e
