"""
Computations on the manifold of
symmetric positive definite matrices.
"""

import numpy as np

EPSILON_SYMMETRIC = 1e-12
EPSILON = 1e-6


def is_symmetric(mat, epsilon=EPSILON):
    """
    Check if a matrix is symmetric.
    """
    # TODO(nina): how about symmetric and posdef?
    difference = mat - mat.transpose()
    if np.linalg.norm(difference) > epsilon:
        print('Matrix not symmetric: {}.'.format(mat))
        return False
    return True


# TODO(nina): The manifold of sym matrices is not a Lie group. Use group exp?
def group_exp(sym_mat):
    """
    Group exponential of the Lie group of
    all invertible matrices has a straight-forward
    computation for symmetric positive definite matrices.
    """
    assert is_symmetric(sym_mat)
    # Make it fully symmetric for numerics.
    sym_mat = (sym_mat + sym_mat.transpose()) / 2

    [diag_elements, vectors] = np.linalg.eigh(sym_mat)

    diag_exp = np.diag(np.exp(diag_elements))
    exp = np.dot(np.dot(vectors, diag_exp), vectors.transpose())

    return(exp)


def group_log(sym_mat):
    """
    Group logarithm of the Lie group of
    all invertible matrices has a straight-forward
    computation for symmetric positive definite matrices.
    """
    assert is_symmetric(sym_mat)
    # Make it fully symmetric for numerics.
    sym_mat = (sym_mat + sym_mat.transpose()) / 2

    [diag_elements, vectors] = np.linalg.eigh(sym_mat)

    for i in range(0, len(diag_elements)):
        if diag_elements[i] <= 0:
            return('Input matrix is not SPdiag_elements')

    diag_log = np.diag(np.log(diag_elements))
    log = np.dot(np.dot(vectors, diag_log), vectors.transpose())

    return log


def square_root(sym_mat):
    assert is_symmetric(sym_mat)
    # Make it fully symmetric for numerics.
    sym_mat = (sym_mat + sym_mat.transpose()) / 2

    [diag_elements, vectors] = np.linalg.eigh(sym_mat)
    for i in range(0, len(diag_elements)):
        if diag_elements[i] <= 0:
            return('Error Input matrix is not SPdiag_elements')

    sqrt_diag = np.diag(np.sqrt(diag_elements))
    sqrt_mat = np.dot(np.dot(vectors, sqrt_diag), vectors.transpose())

    return(sqrt_mat)


def inv_square_root(sym_mat):
    assert is_symmetric(sym_mat)
    # Make it fully symmetric for numerics.
    sym_mat = (sym_mat + sym_mat.transpose()) / 2

    [diag_elements, vectors] = np.linalg.eigh(sym_mat)
    for i in range(0, len(diag_elements)):
        if diag_elements[i] <= 0:
            return('Error Input matrix is not SPdiag_elements')

    inv_sqrt_diag = np.diag(1. / np.sqrt(diag_elements))
    inv_sqrt_mat = np.dot(np.dot(vectors, inv_sqrt_diag), vectors.transpose())

    return(inv_sqrt_mat)


def matrix_to_vectortor(matrix):
    n = size(matrix[:, 0])
    vector = np.zeros(int(n * (n + 1) / 2)).transpose()
    idx = 0
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                vector[idx] = matrix[j, i]
            else:
                vector[idx] = matrix[j, i] / np.sqrt(2.)
            idx = idx + 1
    return vector


def vector_to_matrix(vector):
    N = size(vector[:, 0])
    n = np.int((np.sqrt(8 * N + 1) - 1) / 2)
    matrix = np.zeros((n, n))
    idx = 0
    for j in range(n):
        for i in range(j+1):
            if i == j:
                matrix[i, j] = vector[idx]
            else:
                matrix[i, j] = vector[idx] / np.sqrt(2)
            idx = idx + 1
    for j in range(n):
        for i in range(j + 1, n):
            matrix[i, j] = matrix[j, i]
    return matrix


def riemannian_inner_product(ref_point, tangent_vec_a, tangent_vec_2):
    inv_sqrt_ref_point = inv_square_root(ref_point)
    inv_ref_point = np.linalg.inv(ref_point)

    aux_1 = np.dot(inv_sqrt_ref_point, tangent_vec_a)
    aux_2 = np.dot(aux_1, inv_ref_point)
    aux_3 = np.dot(aux_2, tangent_vec_2)
    inner_product = np.trace(np.dot(aux_3, inv_sqrt_ref_point))

    return inner_product


def riemannian_exp(ref_point, tangent_vec):
    inv_sqrt_ref_point = inv_square_root(ref_point)
    sqrt_ref_point = square_root(ref_point)

    aux_1 = np.dot(inv_sqrt_ref_point, tangent_vec)
    aux_2 = np.dot(aux_1, inv_sqrt_ref_point)
    aux_3 = group_exp(aux_2)
    aux_4 = np.dot(sqrt_ref_point, aux_3)

    riem_exp = np.dot(aux_4, sqrt_ref_point)

    return riem_exp


def riemannian_log(ref_point, point):
    sqrt_ref_point = square_root(ref_point)
    inv_sqrt_ref_point = inv_square_root(ref_point)

    aux_1 = np.dot(np.dot(inv_sqrt_ref_point, point), inv_sqrt_ref_point)
    aux_2 = group_log(aux_1)

    riem_log = np.dot(np.dot(sqrt_ref_point, aux_2), sqrt_ref_point)

    return riem_log


def riemannian_mean(sym_matrices, n_max_iterations, epsilon=EPSILON):
    dists_between_iterates = []
    n_sym_matrices = len(sym_matrices)

    # Initialization of the first iterate with the first tensor
    riem_mean = sym_matrices[0]

    dim = len(riem_mean)

    dist = inf
    it = 0
    while it < n_max_iterations and dist > epsilon:
        # First step project all the tensors onto the tangent space
        # and calculate the mean there
        tangent_mean = np.zeros((dim, dim))
        for j in range(n_sym_matrices):
            tangent_mean += riemannian_log(riem_mean, sym_matrices[j])
        tangent_mean = tangent_mean / n_sym_matrices

        # Second step: reprojection of the''mean'' onto the manifold
        riem_mean = riemannian_exp(riem_mean, tangent_mean)

        dist = riemannian_inner_product(riem_mean, tangent_mean, tangent_mean)
        dists_between_iterates.append(dist)

        if it == n_max_iterations:
            print('warning, max number of iterations reached,'
                  'the result may be inaccurate')

        it += 1

    return (riem_mean, dists_between_iterates)
