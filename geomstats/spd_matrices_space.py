"""
Computations on the manifold of
symmetric positive definite matrices.

X. Pennec. A Riemannian Framework for Tensor Computing. (2004).
"""

import numpy as np
import scipy.linalg

from geomstats.embedded_manifold import EmbeddedManifold
from geomstats.general_linear_group import GeneralLinearGroup
from geomstats.riemannian_metric import RiemannianMetric
import geomstats.vectorization as vectorization

EPSILON = 1e-6
TOLERANCE = 1e-12


def is_symmetric(mat, tolerance=TOLERANCE):
    """Check if a matrix is symmetric."""
    mat = vectorization.to_ndarray(mat, to_ndim=3)
    n_mats, _, _ = mat.shape

    mask = np.zeros(n_mats, dtype=bool)
    for i in range(n_mats):
        mask[i] = np.allclose(mat[i], np.transpose(mat[i]),
                              atol=tolerance)

    return mask


def make_symmetric(mat):
    """Make a matrix fully symmetric to avoid numerical issues."""
    mat = vectorization.to_ndarray(mat, to_ndim=3)
    return (mat + np.transpose(mat, axes=(0, 2, 1))) / 2


# TODO(nina): The manifold of sym matrices is not a Lie group.
# Use 'group_exp' and 'group_log'?
def group_exp(sym_mat):
    """
    Group exponential of the Lie group of
    all invertible matrices has a straight-forward
    computation for symmetric positive definite matrices.
    """
    sym_mat = vectorization.to_ndarray(sym_mat, to_ndim=3)
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
    sym_mat = vectorization.to_ndarray(sym_mat, to_ndim=3)
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


class SPDMatricesSpace(EmbeddedManifold):
    def __init__(self, n):
        super(SPDMatricesSpace, self).__init__(
            dimension=int(n * (n + 1) / 2),
            embedding_manifold=GeneralLinearGroup(n=n))
        self.n = n
        self.metric = SPDMetric(n=n)

    def belongs(self, mat, tolerance=TOLERANCE):
        """
        Check if a matrix belongs to the manifold of
        symmetric positive definite matrices.
        """
        mat = vectorization.to_ndarray(mat, to_ndim=3)
        n_mats, mat_dim, _ = mat.shape

        mask_is_symmetric = is_symmetric(mat, tolerance=tolerance)
        eigenvalues = np.zeros((n_mats, mat_dim))
        eigenvalues[mask_is_symmetric] = np.linalg.eigvalsh(
                                              mat[mask_is_symmetric])

        mask_pos_eigenvalues = np.all(eigenvalues > 0)
        return mask_is_symmetric & mask_pos_eigenvalues

    def vector_from_symmetric_matrix(self, mat):
        """
        Convert the symmetric part of a symmetric matrix
        into a vector.
        """
        mat = vectorization.to_ndarray(mat, to_ndim=3)
        assert np.all(is_symmetric(mat))
        mat = make_symmetric(mat)

        _, mat_dim, _ = mat.shape
        vec_dim = int(mat_dim * (mat_dim + 1) / 2)
        vec = np.zeros(vec_dim)

        idx = 0
        for i in range(mat_dim):
            for j in range(i + 1):
                if i == j:
                    vec[idx] = mat[j, j]
                else:
                    vec[idx] = mat[j, i]
                idx += 1

        return vec

    def symmetric_matrix_from_vector(self, vec):
        """
        Convert a vector into a symmetric matrix.
        """
        vec = vectorization.to_ndarray(vec, to_ndim=2)
        _, vec_dim = vec.shape
        mat_dim = int((np.sqrt(8 * vec_dim + 1) - 1) / 2)
        mat = np.zeros((mat_dim,) * 2)

        lower_triangle_indices = np.tril_indices(mat_dim)
        diag_indices = np.diag_indices(mat_dim)

        mat[lower_triangle_indices] = 2 * vec
        mat[diag_indices] = vec

        mat = make_symmetric(mat)
        return mat

    def random_uniform(self, n_samples=1):
        mat = 2 * np.random.rand(n_samples, self.n, self.n) - 1

        spd_mat = group_exp(mat + np.transpose(mat, axes=(0, 2, 1)))
        return spd_mat

    def random_tangent_vec_uniform(self, n_samples=1, base_point=None):
        if base_point is None:
            base_point = np.eye(self.n)

        base_point = vectorization.to_ndarray(base_point, to_ndim=3)
        n_base_points, _, _ = base_point.shape

        assert n_base_points == n_samples or n_base_points == 1

        sqrt_base_point = np.zeros_like(base_point)
        for i in range(n_base_points):
            sqrt_base_point[i] = scipy.linalg.sqrtm(base_point[i])

        tangent_vec_at_id = (2 * np.random.rand(n_samples,
                                                self.n,
                                                self.n)
                             - 1)
        tangent_vec_at_id = (tangent_vec_at_id
                             + np.transpose(tangent_vec_at_id,
                                            axes=(0, 2, 1)))

        tangent_vec = np.matmul(sqrt_base_point, tangent_vec_at_id)
        tangent_vec = np.matmul(tangent_vec, sqrt_base_point)

        return tangent_vec


class SPDMetric(RiemannianMetric):

    def __init__(self, n):
        super(SPDMetric, self).__init__(dimension=int(n * (n + 1) / 2))

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """
        Compute the inner product of tangent_vec_a and tangent_vec_b
        at point base_point using the affine invariant Riemannian metric.
        """
        inv_base_point = np.linalg.inv(base_point)

        aux_a = np.matmul(inv_base_point, tangent_vec_a)
        aux_b = np.matmul(inv_base_point, tangent_vec_b)
        inner_product = np.trace(np.matmul(aux_a, aux_b), axis1=1, axis2=2)
        inner_product = vectorization.to_ndarray(inner_product,
                                                 to_ndim=2, axis=1)
        return inner_product

    def exp(self, tangent_vec, base_point):
        """
        Compute the Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the metric
        defined in inner_product.

        This gives a symmetric positive definite matrix.
        """
        tangent_vec = vectorization.to_ndarray(tangent_vec, to_ndim=3)
        n_tangent_vecs, _, _ = tangent_vec.shape

        base_point = vectorization.to_ndarray(base_point, to_ndim=3)
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
        point = vectorization.to_ndarray(point, to_ndim=3)
        n_points, _, _ = point.shape

        base_point = vectorization.to_ndarray(base_point, to_ndim=3)
        n_base_points, mat_dim, _ = base_point.shape

        assert (n_points == n_base_points
                or n_points == 1
                or n_base_points == 1)

        sqrt_base_point = np.zeros((n_base_points,) + (mat_dim,) * 2)
        for i in range(n_base_points):
            sqrt_base_point[i] = scipy.linalg.sqrtm(base_point[i])

        inv_sqrt_base_point = np.linalg.inv(sqrt_base_point)
        point_near_id = np.matmul(inv_sqrt_base_point, point)
        point_near_id = np.matmul(point_near_id, inv_sqrt_base_point)
        log_at_id = group_log(point_near_id)

        log = np.matmul(sqrt_base_point, log_at_id)
        log = np.matmul(log, sqrt_base_point)

        return log

    def geodesic(self, initial_point, initial_tangent_vec):
        return super(SPDMetric, self).geodesic(
                                      initial_point=initial_point,
                                      initial_tangent_vec=initial_tangent_vec,
                                      point_ndim=2)
