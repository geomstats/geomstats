"""Utility module of reusable algebra routines."""

import geomstats.backend as gs


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
