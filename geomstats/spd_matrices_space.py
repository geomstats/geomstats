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


def is_symmetric(mat, tolerance=TOLERANCE):
    """Check if a matrix is symmetric."""
    return np.allclose(mat, mat.transpose(), atol=tolerance)


def make_symmetric(mat):
    """Make a matrix fully symmetric to avoid numerical issues."""
    return (mat + mat.transpose()) / 2


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


class SPDMatricesSpace(Manifold):
    def __init__(self, dimension):
        super(SPDMatricesSpace, self).__init__(dimension)
        self.metric = SPDMetric(dimension)

    def belongs(self, mat, tolerance=TOLERANCE):
        """
        Check if a matrix belongs to the manifold of
        symmetric positive definite matrices.
        """
        if is_symmetric(mat, tolerance=tolerance):
            eigenvalues = np.linalg.eigvalsh(mat)
            return np.all(eigenvalues > 0)
        return False

    def matrix_to_vector(self, matrix):
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

    def vector_to_matrix(self, vector):
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

    def random_uniform(self):
        mat = 2 * np.random.rand(self.dimension, self.dimension) - 1

        spd_mat = group_exp(mat + mat.transpose())
        return spd_mat


class SPDMetric(RiemannianMetric):
    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """
        Compute the inner product of tangent_vec_a and tangent_vec_b
        at point base_point using the affine invariant Riemannian metric.
        """
        assert self.belongs(base_point)

        inv_base_point = np.linalg.inv(base_point)

        aux_a = np.dot(inv_base_point, tangent_vec_a)
        aux_b = np.dot(inv_base_point, tangent_vec_b)

        inner_product = np.trace(np.dot(aux_a, aux_b))

        return inner_product

    def exp(self, tangent_vec, base_point):
        """
        Compute the Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the metric
        defined in inner_product.

        This gives a symmetric positive definite matrix.
        """
        sqrt_base_point = scipy.linalg.sqrtm(base_point)
        inv_sqrt_base_point = np.linalg.inv(sqrt_base_point)

        tangent_vec_at_id = np.dot(np.dot(inv_sqrt_base_point,
                                          tangent_vec),
                                   inv_sqrt_base_point)
        exp_from_id = group_exp(tangent_vec_at_id)

        exp = np.dot(sqrt_base_point,
                     np.dot(exp_from_id,
                            sqrt_base_point))

        return exp

    def log(self, point, base_point):
        """
        Compute the Riemannian logarithm at point base_point,
        of point wrt the metric defined in
        inner_product.

        This gives a tangent vector at point base_point.
        """
        sqrt_base_point = scipy.linalg.sqrtm(base_point)
        inv_sqrt_base_point = np.linalg.inv(sqrt_base_point)

        point_near_id = np.dot(np.dot(inv_sqrt_base_point,
                                      point),
                               inv_sqrt_base_point)
        log_at_id = group_log(point_near_id)

        log = np.dot(np.dot(sqrt_base_point,
                            log_at_id),
                     sqrt_base_point)

        return log
