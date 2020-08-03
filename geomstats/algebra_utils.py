"""Utility module of reusable algebra routines."""

import geomstats.backend as gs


EPSILON = 1e-8


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


def taylor_exp(point, taylor_coefs, function, order=7, tol=EPSILON):
    return gs.where(
        point < tol,
        gs.einsum(
            'k,k...->...',
            gs.array(taylor_coefs[:order]),
            gs.array([point ** k for k in range(order)])),
        function(point))
