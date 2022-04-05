"""Utility module of reusable algebra routines."""
import math

import geomstats.backend as gs

EPSILON = 1e-6
COS_TAYLOR_COEFFS = [
    1.0,
    -1.0 / math.factorial(2),
    +1.0 / math.factorial(4),
    -1.0 / math.factorial(6),
    +1.0 / math.factorial(8),
]
SINC_TAYLOR_COEFFS = [
    1.0,
    -1.0 / math.factorial(3),
    +1.0 / math.factorial(5),
    -1.0 / math.factorial(7),
    +1.0 / math.factorial(9),
]
INV_SINC_TAYLOR_COEFFS = [1, 1.0 / 6.0, 7.0 / 360.0, 31.0 / 15120.0, 127.0 / 604800.0]
INV_TANC_TAYLOR_COEFFS = [1.0, -1.0 / 3.0, -1.0 / 45.0, -2.0 / 945.0, -1.0 / 4725.0]
COSC_TAYLOR_COEFFS = [
    1.0 / 2.0,
    -1.0 / math.factorial(4),
    +1.0 / math.factorial(6),
    -1.0 / math.factorial(8),
    +1.0 / math.factorial(10),
]
VAR_INV_TAN_TAYLOR_COEFFS = [1.0 / 12.0, 1.0 / 720.0, 1.0 / 30240.0, 1.0 / 1209600.0]
SINHC_TAYLOR_COEFFS = [
    1.0,
    1 / math.factorial(3),
    1 / math.factorial(5),
    1 / math.factorial(7),
    1 / math.factorial(9),
]
COSH_TAYLOR_COEFFS = [
    1.0,
    1 / math.factorial(2),
    1 / math.factorial(4),
    1 / math.factorial(6),
    1 / math.factorial(8),
]
INV_SINHC_TAYLOR_COEFFS = [
    1.0,
    -1.0 / 6.0,
    7.0 / 360.0,
    -31.0 / 15120.0,
    127.0 / 604800.0,
]
INV_TANH_TAYLOR_COEFFS = [1.0, 1.0 / 3.0, -1.0 / 45.0, 2.0 / 945.0, -1.0 / 4725.0]
ARCTANH_CARD_TAYLOR_COEFFS = [1.0, 1.0 / 3.0, 1.0 / 5.0, 1 / 7.0, 1.0 / 9]


cos_close_0 = {"function": gs.cos, "coefficients": COS_TAYLOR_COEFFS}
sinc_close_0 = {"function": lambda x: gs.sin(x) / x, "coefficients": SINC_TAYLOR_COEFFS}
inv_sinc_close_0 = {
    "function": lambda x: x / gs.sin(x),
    "coefficients": INV_SINC_TAYLOR_COEFFS,
}
inv_tanc_close_0 = {
    "function": lambda x: x / gs.tan(x),
    "coefficients": INV_TANC_TAYLOR_COEFFS,
}
cosc_close_0 = {
    "function": lambda x: (1 - gs.cos(x)) / x**2,
    "coefficients": COSC_TAYLOR_COEFFS,
}
var_sinc_close_0 = {
    "function": lambda x: (x - gs.sin(x)) / x**3,
    "coefficients": [-k for k in SINC_TAYLOR_COEFFS[1:]],
}
var_inv_tanc_close_0 = {
    "function": lambda x: (1 - (x / gs.tan(x))) / x**2,
    "coefficients": VAR_INV_TAN_TAYLOR_COEFFS,
}
sinch_close_0 = {
    "function": lambda x: gs.sinh(x) / x,
    "coefficients": SINHC_TAYLOR_COEFFS,
}
cosh_close_0 = {"function": gs.cosh, "coefficients": COSH_TAYLOR_COEFFS}
inv_sinch_close_0 = {
    "function": lambda x: x / gs.sinh(x),
    "coefficients": INV_SINHC_TAYLOR_COEFFS,
}
inv_tanh_close_0 = {
    "function": lambda x: x / gs.tanh(x),
    "coefficients": INV_TANH_TAYLOR_COEFFS,
}
arctanh_card_close_0 = {
    "function": lambda x: gs.arctanh(x) / x,
    "coefficients": ARCTANH_CARD_TAYLOR_COEFFS,
}


def from_vector_to_diagonal_matrix(vector, num_diag=0):
    """Create diagonal matrices from rows of a matrix.

    Parameters
    ----------
    vector : array-like, shape=[m, n]
    num_diag : int
        number of diagonal in result matrix. If 0, the result matrix is a
        diagonal matrix; if positive, the result matrix has an upper-right
        non-zero diagonal; if negative, the result matrix has a lower-left
        non-zero diagonal.
        Optional, Default: 0.

    Returns
    -------
    diagonals : array-like, shape=[m, n, n]
        3-dimensional array where the `i`-th n-by-n array `diagonals[i, :, :]`
        is a diagonal matrix containing the `i`-th row of `vector`.
    """
    num_columns = gs.shape(vector)[-1]
    identity = gs.eye(num_columns)
    identity = gs.cast(identity, vector.dtype)
    diagonals = gs.einsum("...i,ij->...ij", vector, identity)
    diagonals = gs.to_ndarray(diagonals, to_ndim=3)
    num_lines = diagonals.shape[0]
    if num_diag > 0:
        left_zeros = gs.zeros((num_lines, num_columns, num_diag))
        lower_zeros = gs.zeros((num_lines, num_diag, num_columns + num_diag))
        diagonals = gs.concatenate((left_zeros, diagonals), axis=2)
        diagonals = gs.concatenate((diagonals, lower_zeros), axis=1)
    elif num_diag < 0:
        num_diag = gs.abs(num_diag)
        right_zeros = gs.zeros((num_lines, num_columns, num_diag))
        upper_zeros = gs.zeros((num_lines, num_diag, num_columns + num_diag))
        diagonals = gs.concatenate((diagonals, right_zeros), axis=2)
        diagonals = gs.concatenate((upper_zeros, diagonals), axis=1)
    return gs.squeeze(diagonals) if gs.ndim(vector) == 1 else diagonals


def taylor_exp_even_func(point, taylor_function, order=5, tol=EPSILON):
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
        "k,k...->...",
        gs.array(taylor_function["coefficients"][:order]),
        gs.array([point**k for k in range(order)]),
    )
    point_ = gs.where(gs.abs(point) <= tol, tol, point)
    exact = taylor_function["function"](gs.sqrt(point_))
    result = gs.where(gs.abs(point) < tol, approx, exact)
    return result


def flip_determinant(matrix, det):
    """Change sign of the determinant if it is negative.

    For a batch of matrices, multiply the matrices which have negative
    determinant by a diagonal matrix :math: `diag(1,...,1,-1) from the right.
    This changes the sign of the last column of the matrix.

    Parameters
    ----------
    matrix : array-like, shape=[...,n ,m]
        Matrix to transform.

    det : array-like, shape=[...]
        Determinant of matrix, or any other scalar to use as threshold to
        determine whether to change the sign of the last column of matrix.

    Returns
    -------
    matrix_flipped : array-like, shape=[..., n, m]
        Matrix with the sign of last column changed if det < 0.
    """
    if gs.any(det < 0):
        ones = gs.ones(matrix.shape[-1])
        reflection_vec = gs.concatenate([ones[:-1], gs.array([-1.0])], axis=0)
        mask = gs.cast(det < 0, matrix.dtype)
        sign = mask[..., None] * reflection_vec + (1.0 - mask)[..., None] * ones
        return gs.einsum("...ij,...j->...ij", matrix, sign)
    return matrix


def rotate_points(points, end_point):
    """Apply to points the rotation from north_pole to end_point.

    A QR decomposition is used to find the rotation that maps the north pole
    (1, 0,...,0) to the end_point, then this rotation is applied to the
    input points.

    Parameters
    ----------
    points : array-like, shape=[..., n]
        Points to rotate.
    end_point : array-like, shape=[n, ]
        Point to parametrise the rotation.

    Returns
    -------
    rotated_points : array-like, shape=[..., n]
        Points after the rotation.
    """
    n = end_point.shape[0]
    base_point = gs.array([1.0] + [0] * (n - 1))
    embedded = gs.concatenate([end_point[None, :], gs.zeros((n - 1, n))])
    norm = gs.linalg.norm(end_point)
    q, _ = gs.linalg.qr(gs.transpose(embedded) / norm)
    new_points = gs.matmul(points[None, :], gs.transpose(q)) * norm
    if not gs.allclose(gs.matmul(q, base_point[:, None])[:, 0], end_point):
        new_points = -new_points
    return new_points[0]
