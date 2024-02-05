"""Numpy based linear algebra backend."""

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
from scipy.linalg import expm

from .._shared_numpy.linalg import (
    fractional_matrix_power,
    is_single_matrix_pd,
    logm,
    qr,
    quadratic_assignment,
    solve_sylvester,
    sqrtm,
)
