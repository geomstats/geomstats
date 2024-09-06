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
    solve,
    solve_sylvester,
    sqrtm,
)
