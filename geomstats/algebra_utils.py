"""Utility module of reusable algebra routines."""
import math

import geomstats.backend as gs


EPSILON = 1e-6
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
INV_SINC_TAYLOR_COEFFS = [1, 1. / 6., 7. / 360., 31. / 15120., 127. / 604800.]
INV_TANC_TAYLOR_COEFFS = [1., - 1. / 3., - 1. / 45., - 2. / 945., - 1. / 4725.]
COSC_TAYLOR_COEFFS = [1. / 2.,
                      - 1.0 / math.factorial(4),
                      + 1.0 / math.factorial(6),
                      - 1.0 / math.factorial(8),
                      + 1. / math.factorial(10)]
VAR_INV_TAN_TAYLOR_COEFFS = [
    1. / 12., 1. / 720., 1. / 30240., 1. / 1209600.]

cos_close_0 = {'function': gs.cos, 'coeffs': COS_TAYLOR_COEFFS}
sinc_close_0 = {
    'function': lambda x: gs.sin(x) / x, 'coeffs': SINC_TAYLOR_COEFFS}
inv_sinc_close_0 = {
    'function': lambda x: x / gs.sin(x), 'coeffs': INV_SINC_TAYLOR_COEFFS}
inv_tanc_close_0 = {
    'function': lambda x: x / gs.tan(x), 'coeffs': INV_TANC_TAYLOR_COEFFS}
cosc_close_0 = {
    'function': lambda x: (1 - gs.cos(x)) / x ** 2,
    'coeffs': COSC_TAYLOR_COEFFS}
var_inv_tanc_close_0 = {
    'function': lambda x: (1 - (x / gs.tan(x))) / x ** 2,
    'coeffs': VAR_INV_TAN_TAYLOR_COEFFS}


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
        point, taylor_function, order=5, tol=EPSILON):
    """Taylor Approximation of an even function around zero.

    Parameters
    ----------
    point
    taylor_function
    order
    tol

    Returns
    -------

    """
    approx = gs.einsum(
        'k,k...->...', gs.array(taylor_function['coeffs'][:order]),
        gs.array([point ** k for k in range(order)]))
    point_ = gs.where(gs.abs(point) <= tol, tol, point)
    exact = taylor_function['function'](gs.sqrt(point_))
    result = gs.where(gs.abs(point) < tol, approx, exact)
    return result
