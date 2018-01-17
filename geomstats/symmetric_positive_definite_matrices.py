"""
Computations on the manifold of
symmetric positive definite matrices.

X. Pennec. A Riemannian Framework for Tensor Computing. (2004).
"""

import logging
import math
import numpy as np
import scipy.linalg

TOL_SYMMETRIC = 1e-12
EPSILON = 1e-6


def is_symmetric(mat, tolerance=TOL_SYMMETRIC):
    """Check if a matrix is symmetric."""
    return np.allclose(mat, mat.transpose(), atol=tolerance)


def make_symmetric(mat):
    """Make a matrix fully symmetric to avoid numerical issues."""
    return (mat + mat.transpose()) / 2


def belongs(mat, tolerance=TOL_SYMMETRIC):
    """
    Check if a matrix belongs to the manifold of
    symmetric positive definite matrices.
    """
    if is_symmetric(mat, tolerance=tolerance):
        eigenvalues = np.linalg.eigvalsh(mat)
        return np.all(eigenvalues > 0)
    return False


# TODO(nina): The manifold of sym matrices is not a Lie group.
# Use 'group_exp' and 'group_log'?
def group_exp(sym_mat):
    """
    Group exponential of the Lie group of
    all invertible matrices has a straight-forward
    computation for symmetric positive definite matrices.
    """
    assert is_symmetric(sym_mat)
    sym_mat = make_symmetric(sym_mat)

    [eigenvalues, vectors] = np.linalg.eigh(sym_mat)

    diag_exp = np.diag(np.exp(eigenvalues))
    exp = np.dot(np.dot(vectors, diag_exp), vectors.transpose())

    return(exp)


def group_log(sym_mat):
    """
    Group logarithm of the Lie group of
    all invertible matrices has a straight-forward
    computation for symmetric positive definite matrices.
    """
    assert is_symmetric(sym_mat)
    sym_mat = make_symmetric(sym_mat)

    [eigenvalues, vectors] = np.linalg.eigh(sym_mat)

    assert np.all(eigenvalues > 0)

    diag_log = np.diag(np.log(eigenvalues))
    log = np.dot(np.dot(vectors, diag_log), vectors.transpose())

    return log


def matrix_to_vector(matrix):
    """
    Convert the symmetric part of a symmetric matrix
    into a vector.
    """
    # TODO(nina): why factor np.sqrt(2)
    assert is_symmetric(matrix)
    matrix = make_symmetric(matrix)

    dim_mat, _ = matrix.shape
    dim_vec = int(dim_mat * (dim_mat + 1) / 2)
    vector = np.zeros(dim_vec)

    idx = 0
    for i in range(dim_mat):
        for j in range(i + 1):
            if i == j:
                vector[idx] = matrix[j, j]
            else:
                vector[idx] = matrix[j, i] * np.sqrt(2.)
            idx += 1

    return vector


def vector_to_matrix(vector):
    """
    Convert a vector into a symmetric matrix.
    """
    # TODO(nina): why factor np.sqrt(2)
    dim_vec = len(vector)
    dim_mat = int((np.sqrt(8 * dim_vec + 1) - 1) / 2)
    matrix = np.zeros((dim_mat, dim_mat))

    lower_triangle_indices = np.tril_indices(dim_mat)
    diag_indices = np.diag_indices(dim_mat)

    matrix[lower_triangle_indices] = 2 * vector / np.sqrt(2)
    matrix[diag_indices] = vector

    matrix = make_symmetric(matrix)
    return matrix


def riemannian_inner_product(ref_point, tangent_vec_a, tangent_vec_b):
    """
    Compute the inner product of tangent_vec_a and tangent_vec_b
    at point ref_point using the affine invariant Riemannian metric.
    """
    inv_ref_point = np.linalg.inv(ref_point)

    aux_a = np.dot(inv_ref_point, tangent_vec_a)
    aux_b = np.dot(inv_ref_point, tangent_vec_b)

    inner_product = np.trace(np.dot(aux_a, aux_b))

    return inner_product


def riemannian_exp(ref_point, tangent_vec):
    """
    Compute the Riemannian exponential at point ref_point
    of tangent vector tangent_vec wrt the metric
    defined in riemannian_inner_product.

    This gives a symmetric positive definite matrix.
    """
    sqrt_ref_point = scipy.linalg.sqrtm(ref_point)
    inv_sqrt_ref_point = np.linalg.inv(sqrt_ref_point)

    tangent_vec_at_id = np.dot(np.dot(inv_sqrt_ref_point,
                                      tangent_vec),
                               inv_sqrt_ref_point)
    riem_exp_from_id = group_exp(tangent_vec_at_id)

    riem_exp = np.dot(sqrt_ref_point, np.dot(riem_exp_from_id, sqrt_ref_point))

    return riem_exp


def riemannian_log(ref_point, point):
    """
    Compute the Riemannian logarithm at point ref_point,
    of point wrt the metric defined in
    riemannian_inner_product.

    This gives a tangent vector at point ref_point.
    """
    sqrt_ref_point = scipy.linalg.sqrtm(ref_point)
    inv_sqrt_ref_point = np.linalg.inv(sqrt_ref_point)

    point_near_id = np.dot(np.dot(inv_sqrt_ref_point,
                                  point),
                           inv_sqrt_ref_point)
    riem_log_at_id = group_log(point_near_id)

    riem_log = np.dot(np.dot(sqrt_ref_point, riem_log_at_id), sqrt_ref_point)

    return riem_log


def riemannian_mean(sym_matrices, n_max_iterations, epsilon=EPSILON):
    """
    Compute the Riemannian mean (Frechet mean) iterating 3 steps:
    - Project all the matrices onto the tangent space using the riemannian log
    - Calculate the tangent mean on the tangent space
    - Shoot the tangent mean onto the manifold using the riemannian exp

    Initialization with one of the matrices.
    """
    # TODO(nina): profile this code to study performance
    dists_between_iterates = []
    n_sym_matrices = len(sym_matrices)

    riem_mean = sym_matrices[0]

    dim = len(riem_mean)

    dist = math.inf
    it = 0
    while it < n_max_iterations and dist > epsilon:
        tangent_mean = np.zeros((dim, dim))
        for j in range(n_sym_matrices):
            tangent_mean += riemannian_log(riem_mean, sym_matrices[j])
        tangent_mean = tangent_mean / n_sym_matrices

        riem_mean = riemannian_exp(riem_mean, tangent_mean)

        dist = riemannian_inner_product(riem_mean, tangent_mean, tangent_mean)
        dists_between_iterates.append(dist)

        if it == n_max_iterations:
            logging.warning('Maximum number of iterations {} reached.'
                            'The riemannian_mean may be inaccurate'
                            ''.format(n_max_iterations))

        it += 1

    return (riem_mean, dists_between_iterates)
