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
    matrix_power,
    matrix_rank,
    norm,
    svd,
)
from scipy.linalg import expm

from .._shared_numpy.linalg import (
    fractional_matrix_power,
    is_single_matrix_pd,
    logm,
    polar,
    qr,
    quadratic_assignment,
    solve_sylvester,
    sqrtm,
)


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
