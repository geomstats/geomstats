"""Utility module of reusable algebra routines."""
import math

import geomstats.backend as gs


EPSILON = 1e-4
COS_TAYLOR_COEFFS = [1.,
                     - 1.0 / math.factorial(2),
                     + 1.0 / math.factorial(4),
                     - 1.0 / math.factorial(6),
                     + 1.0 / math.factorial(8)]
SINC_TAYLOR_COEFFS = [1.,
                      - 1.0 / math.factorial(3),
                      + 1.0 / math.factorial(5),
                      - 1.0 / math.factorial(7),
                      + 1.0 / math.factorial(9)]
INV_SINC_TAYLOR_COEFFS = [1,
                          1. / 6.,
                          7. / 360.,
                          31. / 15120.,
                          127. / 604800.]
INV_TANC_TAYLOR_COEFFS = [1.,
                          - 1. / 3.,
                          - 1. / 45.,
                          - 2. / 945.,
                          - 1. / 4725.]


def from_vector_to_diagonal_matrix(vector):
    """Create diagonal matrices from rows of a matrix.

    Parameters
    ----------
    vector : array-like, shape=[m, n]

    Returns
    -------
    diagonals : array-like, shape=[m, n, n]
        3-dimensional array where the `i`-th n-by-n array `diagonals[i, :, :]`
        is a diagonal matrix containing the `i`-th row of `vector`.
    """
    num_columns = gs.shape(vector)[-1]
    identity = gs.eye(num_columns)
    identity = gs.cast(identity, vector.dtype)
    diagonals = gs.einsum('...i,ij->...ij', vector, identity)
    return diagonals


def taylor_exp_even_func(
        point, taylor_coefs, function, order=7, tol=EPSILON):
    """

    Parameters
    ----------
    point
    taylor_coefs
    function
    order
    tol
    even: bool
        If True, the Taylor approximation is composed with the square
        function, so point is considered point^2

    Returns
    -------

    """
    approx = gs.einsum(
        'k,k...->...', gs.array(taylor_coefs[:order]),
        gs.array([point ** k for k in range(order)]))
    point_ = gs.where(gs.abs(point) <= tol, tol, point)
    exact = function(gs.sqrt(point_))
    result = gs.where(gs.abs(point) < tol, approx, exact)
    return result
