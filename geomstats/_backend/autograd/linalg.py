"""Autograd based linear algebra backend."""

import functools as _functools

import autograd.numpy as _np
from autograd.extend import defvjp as _defvjp
from autograd.extend import primitive as _primitive
from autograd.numpy.linalg import (
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
from autograd.scipy.linalg import expm

from .._shared_numpy.linalg import fractional_matrix_power, is_single_matrix_pd
from .._shared_numpy.linalg import logm as _logm
from .._shared_numpy.linalg import qr, quadratic_assignment, solve_sylvester, sqrtm


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


logm = _primitive(_logm)

_logm_vjp = _functools.partial(_adjoint, fn=logm)
_defvjp(logm, _logm_vjp)
