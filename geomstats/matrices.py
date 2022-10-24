"""Util functions for matrix operations"""
import logging
from functools import reduce

import geomstats.backend as gs
from geomstats.algebra import flip_determinant, from_vector_to_diagonal_matrix


def equal(mat_a, mat_b, atol=gs.atol):
    """Test if matrices a and b are close.

    Parameters
    ----------
    mat_a : array-like, shape=[..., dim1, dim2]
        Matrix.
    mat_b : array-like, shape=[..., dim2, dim3]
        Matrix.
    atol : float
        Tolerance.
        Optional, default: backend atol.

    Returns
    -------
    eq : array-like, shape=[...,]
        Boolean evaluating if the matrices are close.
    """
    return gs.all(gs.isclose(mat_a, mat_b, atol=atol), (-2, -1))

def mul(*args):
    """Compute the product of matrices a1, ..., an.

    Parameters
    ----------
    a1 : array-like, shape=[..., dim_1, dim_2]
        Matrix.
    a2 : array-like, shape=[..., dim_2, dim_3]
        Matrix.
    ...
    an : array-like, shape=[..., dim_n-1, dim_n]
        Matrix.

    Returns
    -------
    mul : array-like, shape=[..., dim_1, dim_n]
        Result of the product of matrices.
    """
    return reduce(gs.matmul, args)

def transpose(mat):
    """Return the transpose of matrices.

    Parameters
    ----------
    mat : array-like, shape=[..., n, n]
        Matrix.

    Returns
    -------
    transpose : array-like, shape=[..., n, n]
        Transposed matrix.
    """
    is_vectorized = gs.ndim(mat) == 3
    axes = (0, 2, 1) if is_vectorized else (1, 0)
    return gs.transpose(mat, axes)

def diagonal(mat):
    """Return the diagonal of a matrix as a vector.

    Parameters
    ----------
    mat : array-like, shape=[..., m, n]
        Matrix.

    Returns
    -------
    diagonal : array-like, shape=[..., min(m, n)]
        Vector of diagonal coefficients.
    """
    return gs.diagonal(mat, axis1=-2, axis2=-1)

def is_square(mat):
    """Check if a matrix is square.

    Parameters
    ----------
    mat : array-like, shape=[..., m, n]
        Matrix.

    Returns
    -------
    is_square : array-like, shape=[...,]
        Boolean evaluating if the matrix is square.
    """
    n = mat.shape[-1]
    m = mat.shape[-2]
    return m == n

def frobenius_product(mat_1, mat_2):
    """Compute Frobenius inner-product of two matrices.

    The `einsum` function is used to avoid computing a matrix product. It
    is also faster than using a sum an element-wise product.

    Parameters
    ----------
    mat_1 : array-like, shape=[..., m, n]
        Matrix.
    mat_2 : array-like, shape=[..., m, n]
        Matrix.

    Returns
    -------
    product : array-like, shape=[...,]
        Frobenius inner-product of mat_1 and mat_2
    """
    return gs.einsum("...ij,...ij->...", mat_1, mat_2)

def trace_product(mat_1, mat_2):
    """Compute trace of the product of two matrices.

    The `einsum` function is used to avoid computing a matrix product. It
    is also faster than using a sum an element-wise product.

    Parameters
    ----------
    mat_1 : array-like, shape=[..., m, n]
        Matrix.
    mat_2 : array-like, shape=[..., m, n]
        Matrix.

    Returns
    -------
    product : array-like, shape=[...,]
        Frobenius inner-product of mat_1 and mat_2
    """
    return gs.einsum("...ij,...ji->...", mat_1, mat_2)
