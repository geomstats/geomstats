"""Autograd based linear algebra backend."""

import autograd.numpy as _np
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
from autograd.scipy.linalg import expm, fractional_matrix_power, sqrtm

from ..numpy.linalg import logm as _logm
from ..numpy.linalg import quadratic_assignment  # NOQA
from ..numpy.linalg import solve_sylvester as _solve_sylvester

logm = _primitive(_logm)
solve_sylvester = _primitive(_solve_sylvester)


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
