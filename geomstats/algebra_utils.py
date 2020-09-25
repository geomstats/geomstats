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
SINHC_TAYLOR_COEFFS = [1.,
                       1 / math.factorial(3),
                       1 / math.factorial(5),
                       1 / math.factorial(7),
                       1 / math.factorial(9)]
COSH_TAYLOR_COEFFS = [1.,
                      1 / math.factorial(2),
                      1 / math.factorial(4),
                      1 / math.factorial(6),
                      1 / math.factorial(8)]
INV_SINHC_TAYLOR_COEFFS = [
    1., - 1. / 6., 7. / 360., - 31. / 15120., 127. / 604800.]
INV_TANH_TAYLOR_COEFFS = [1., 1. / 3., - 1. / 45., 2. / 945., -1. / 4725.]


cos_close_0 = {'function': gs.cos, 'coefficients': COS_TAYLOR_COEFFS}
sinc_close_0 = {
    'function': lambda x: gs.sin(x) / x, 'coefficients': SINC_TAYLOR_COEFFS}
inv_sinc_close_0 = {
    'function': lambda x: x / gs.sin(x),
    'coefficients': INV_SINC_TAYLOR_COEFFS}
inv_tanc_close_0 = {
    'function': lambda x: x / gs.tan(x),
    'coefficients': INV_TANC_TAYLOR_COEFFS}
cosc_close_0 = {
    'function': lambda x: (1 - gs.cos(x)) / x ** 2,
    'coefficients': COSC_TAYLOR_COEFFS}
var_sinc_close_0 = {
    'function': lambda x: (x - gs.sin(x)) / x ** 3,
    'coefficients': [-k for k in SINC_TAYLOR_COEFFS[1:]]}
var_inv_tanc_close_0 = {
    'function': lambda x: (1 - (x / gs.tan(x))) / x ** 2,
    'coefficients': VAR_INV_TAN_TAYLOR_COEFFS}
sinch_close_0 = {
    'function': lambda x: gs.sinh(x) / x,
    'coefficients': SINHC_TAYLOR_COEFFS}
cosh_close_0 = {'function': gs.cosh, 'coefficients': COSH_TAYLOR_COEFFS}
inv_sinch_close_0 = {
    'function': lambda x: x / gs.sinh(x),
    'coefficients': INV_SINHC_TAYLOR_COEFFS}
inv_tanh_close_0 = {
    'function': lambda x: x / gs.tanh(x),
    'coefficients': INV_TANH_TAYLOR_COEFFS}


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
    point : array-like
        Argument of the function to approximate.
    taylor_function : dict with following keys
        function : callable
            Even function to approximate around zero.
        coefficients : list
            Taylor coefficients of even order at zero.
    order : int
        Order of the Taylor approximation.
        Optional, Default: 5.
    tol : float
        Threshold to use the approximation instead of the function's value.
        Where `abs(point) <= tol`, the approximation is returned.

    Returns
    -------
    function_value: array-like
        Value of the function at point.
    """
    approx = gs.einsum(
        'k,k...->...', gs.array(taylor_function['coefficients'][:order]),
        gs.array([point ** k for k in range(order)]))
    point_ = gs.where(gs.abs(point) <= tol, tol, point)
    exact = taylor_function['function'](gs.sqrt(point_))
    result = gs.where(gs.abs(point) < tol, approx, exact)
    return result
