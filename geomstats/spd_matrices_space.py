"""
Computations on the manifold of
symmetric positive definite matrices.

X. Pennec. A Riemannian Framework for Tensor Computing. (2004).
"""

import numpy as np
import scipy.linalg

from geomstats.manifold import Manifold
from geomstats.riemannian_metric import RiemannianMetric


EPSILON = 1e-6
TOLERANCE = 1e-12

# TODO(nina): refactor use of np.expand_dims that appears in all modules


def is_symmetric(mat, tolerance=TOLERANCE):
    """Check if a matrix is symmetric."""
    if mat.ndim == 2:
        mat = np.expand_dims(mat, axis=0)
    assert mat.ndim == 3
    n_mats, _, _ = mat.shape

    mask = np.zeros(n_mats, dtype=bool)
    for i in range(n_mats):
        mask[i] = np.allclose(mat[i], np.transpose(mat[i]),
                              atol=tolerance)

    return mask


def make_symmetric(mat):
    """Make a matrix fully symmetric to avoid numerical issues."""
    if mat.ndim == 2:
        mat = np.expand_dims(mat, axis=0)
    assert mat.ndim == 3
    return (mat + np.transpose(mat, axes=(0, 2, 1))) / 2


# TODO(nina): The manifold of sym matrices is not a Lie group.
# Use 'group_exp' and 'group_log'?
def group_exp(sym_mat):
    """
    Group exponential of the Lie group of
    all invertible matrices has a straight-forward
    computation for symmetric positive definite matrices.
    """
    if sym_mat.ndim == 2:
        sym_mat = np.expand_dims(sym_mat, axis=0)
    assert sym_mat.ndim == 3
    n_sym_mats, mat_dim, _ = sym_mat.shape

    assert np.all(is_symmetric(sym_mat))
    sym_mat = make_symmetric(sym_mat)

    [eigenvalues, vectors] = np.linalg.eigh(sym_mat)
    diag_exp = np.zeros((n_sym_mats, mat_dim, mat_dim))
    for i in range(n_sym_mats):
        diag_exp[i] = np.diag(np.exp(eigenvalues[i]))

    exp = np.matmul(diag_exp, np.transpose(vectors, axes=(0, 2, 1)))
    exp = np.matmul(vectors, exp)
    return exp


def group_log(sym_mat):
    """
    Group logarithm of the Lie group of
    all invertible matrices has a straight-forward
    computation for symmetric positive definite matrices.
    """
    if sym_mat.ndim == 2:
        sym_mat = np.expand_dims(sym_mat, axis=0)
    assert sym_mat.ndim == 3
    n_sym_mats, mat_dim, _ = sym_mat.shape

    assert np.all(is_symmetric(sym_mat))
    sym_mat = make_symmetric(sym_mat)
    [eigenvalues, vectors] = np.linalg.eigh(sym_mat)
    assert np.all(eigenvalues > 0)
    diag_log = np.zeros((n_sym_mats, mat_dim, mat_dim))
    for i in range(n_sym_mats):
        diag_log[i] = np.diag(np.log(eigenvalues[i]))

    log = np.matmul(diag_log, np.transpose(vectors, axes=(0, 2, 1)))
    log = np.matmul(vectors, log)
    return log


class SPDMatricesSpace(Manifold):
    def __init__(self, dimension):
        super(SPDMatricesSpace, self).__init__(dimension)
        self.metric = SPDMetric(dimension)

    def belongs(self, mat, tolerance=TOLERANCE):
        """
        Check if a matrix belongs to the manifold of
        symmetric positive definite matrices.
        """
        if mat.ndim == 2:
            mat = np.expand_dims(mat, axis=0)
        assert mat.ndim == 3
        n_mats, mat_size, _ = mat.shape

        mask_is_symmetric = is_symmetric(mat, tolerance=tolerance)
        eigenvalues = np.zeros((n_mats, mat_size))
        eigenvalues[mask_is_symmetric] = np.linalg.eigvalsh(
                                              mat[mask_is_symmetric])

        mask_pos_eigenvalues = np.all(eigenvalues > 0)
        return mask_is_symmetric & mask_pos_eigenvalues

    def matrix_to_vector(self, matrix):
        """
        Convert the symmetric part of a symmetric matrix
        into a vector.
        """
        # TODO(nina): why factor np.sqrt(2)
        if matrix.ndim == 2:
            matrix = np.expand_dims(matrix, axis=0)
        assert matrix.ndim == 3
        assert np.all(is_symmetric(matrix))
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

    def vector_to_matrix(self, vector):
        """
        Convert a vector into a symmetric matrix.
        """
        if vector.ndim == 1:
            vector = np.expand_dims(vector, axis=0)
        assert vector.ndim == 2
        # TODO(nina): do we need factor np.sqrt(2) and why?
        _, vec_dim = vector.shape
        mat_dim = int((np.sqrt(8 * vec_dim + 1) - 1) / 2)
        matrix = np.zeros((mat_dim, mat_dim))

        lower_triangle_indices = np.tril_indices(mat_dim)
        diag_indices = np.diag_indices(mat_dim)

        matrix[lower_triangle_indices] = 2 * vector / np.sqrt(2)
        matrix[diag_indices] = vector

        matrix = make_symmetric(matrix)
        return matrix

    def random_uniform(self, n_samples=1):
        mat = 2 * np.random.rand(n_samples, self.dimension, self.dimension) - 1

        spd_mat = group_exp(mat + np.transpose(mat, axes=(0, 2, 1)))
        return spd_mat

    def random_tangent_vec_uniform(self, n_samples=1, base_point=None):
        if base_point is None:
            base_point = np.eye(self.dimension)

        if base_point.ndim == 2:
            base_point = np.expand_dims(base_point, axis=0)

        n_base_points, _, _ = base_point.shape
        assert n_base_points == n_samples or n_base_points == 1

        sqrt_base_point = np.zeros_like(base_point)
        for i in range(n_base_points):
            sqrt_base_point[i] = scipy.linalg.sqrtm(base_point[i])

        tangent_vec_at_id = (2 * np.random.rand(n_samples,
                                                self.dimension,
                                                self.dimension)
                             - 1)
        tangent_vec_at_id = (tangent_vec_at_id
                             + np.transpose(tangent_vec_at_id,
                                            axes=(0, 2, 1)))

        tangent_vec = np.matmul(sqrt_base_point, tangent_vec_at_id)
        tangent_vec = np.matmul(tangent_vec, sqrt_base_point)

        return tangent_vec


class SPDMetric(RiemannianMetric):
    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """
        Compute the inner product of tangent_vec_a and tangent_vec_b
        at point base_point using the affine invariant Riemannian metric.
        """
        inv_base_point = np.linalg.inv(base_point)

        aux_a = np.matmul(inv_base_point, tangent_vec_a)
        aux_b = np.matmul(inv_base_point, tangent_vec_b)
        inner_product = np.trace(np.matmul(aux_a, aux_b), axis1=1, axis2=2)
        if inner_product.ndim == 1:
            inner_product = np.expand_dims(inner_product, axis=1)
        return inner_product

    def exp(self, tangent_vec, base_point):
        """
        Compute the Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the metric
        defined in inner_product.

        This gives a symmetric positive definite matrix.
        """
        if tangent_vec.ndim == 2:
            tangent_vec = np.expand_dims(tangent_vec, axis=0)
        assert tangent_vec.ndim == 3

        if base_point.ndim == 2:
            base_point = np.expand_dims(base_point, axis=0)
        assert base_point.ndim == 3

        n_tangent_vecs, _, _ = tangent_vec.shape
        n_base_points, mat_dim, _ = base_point.shape

        assert (n_tangent_vecs == n_base_points
                or n_tangent_vecs == 1
                or n_base_points == 1)

        sqrt_base_point = np.zeros((n_base_points, mat_dim, mat_dim))
        for i in range(n_base_points):
            sqrt_base_point[i] = scipy.linalg.sqrtm(base_point[i])

        inv_sqrt_base_point = np.linalg.inv(sqrt_base_point)

        tangent_vec_at_id = np.matmul(inv_sqrt_base_point,
                                      tangent_vec)
        tangent_vec_at_id = np.matmul(tangent_vec_at_id,
                                      inv_sqrt_base_point)
        exp_from_id = group_exp(tangent_vec_at_id)

        exp = np.matmul(exp_from_id, sqrt_base_point)
        exp = np.matmul(sqrt_base_point, exp)

        return exp

    def log(self, point, base_point):
        """
        Compute the Riemannian logarithm at point base_point,
        of point wrt the metric defined in
        inner_product.

        This gives a tangent vector at point base_point.
        """
        if point.ndim == 2:
            point = np.expand_dims(point, axis=0)
        assert point.ndim == 3

        if base_point.ndim == 2:
            base_point = np.expand_dims(base_point, axis=0)
        assert base_point.ndim == 3

        n_points, _, _ = point.shape
        n_base_points, mat_dim, _ = base_point.shape

        assert (n_points == n_base_points
                or n_points == 1
                or n_base_points == 1)

        sqrt_base_point = np.zeros((n_base_points, mat_dim, mat_dim))
        for i in range(n_base_points):
            sqrt_base_point[i] = scipy.linalg.sqrtm(base_point[i])

        inv_sqrt_base_point = np.linalg.inv(sqrt_base_point)
        point_near_id = np.matmul(inv_sqrt_base_point, point)
        point_near_id = np.matmul(point_near_id, inv_sqrt_base_point)
        log_at_id = group_log(point_near_id)

        log = np.matmul(sqrt_base_point, log_at_id)
        log = np.matmul(log, sqrt_base_point)

        return log
