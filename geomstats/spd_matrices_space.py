"""
The manifold of symmetric positive definite (SPD) matrices.
"""

import geomstats.backend as gs

from geomstats.embedded_manifold import EmbeddedManifold
from geomstats.general_linear_group import GeneralLinearGroup
from geomstats.riemannian_metric import RiemannianMetric

EPSILON = 1e-6
TOLERANCE = 1e-12


def is_symmetric(mat, tolerance=TOLERANCE):
    """Check if a matrix is symmetric."""
    mat = gs.to_ndarray(mat, to_ndim=3)
    n_mats, _, _ = mat.shape
    mat_transpose = gs.transpose(mat, axes=(0, 2, 1))

    mask = gs.isclose(mat, mat_transpose, atol=tolerance)
    mask = gs.all(mask, axis=(1, 2))

    return mask


def make_symmetric(mat):
    """Make a matrix fully symmetric to avoid numerical issues."""
    mat = gs.to_ndarray(mat, to_ndim=3)
    return (mat + gs.transpose(mat, axes=(0, 2, 1))) / 2


def sqrtm(sym_mat):
    sym_mat = gs.to_ndarray(sym_mat, to_ndim=3)

    [eigenvalues, vectors] = gs.linalg.eigh(sym_mat)

    sqrt_eigenvalues = gs.sqrt(eigenvalues)

    aux = gs.einsum('ijk,ik->ijk', vectors, sqrt_eigenvalues)
    sqrt_mat = gs.einsum('ijk,ilk->ijl', aux, vectors)

    sqrt_mat = gs.to_ndarray(sqrt_mat, to_ndim=3)
    return sqrt_mat


# TODO(nina): The manifold of sym matrices is not a Lie group.
# Use 'group_exp' and 'group_log'?
def group_exp(sym_mat):
    """
    Group exponential of the Lie group of
    all invertible matrices has a straight-forward
    computation for symmetric positive definite matrices.
    """
    sym_mat = gs.to_ndarray(sym_mat, to_ndim=3)
    n_sym_mats, mat_dim, _ = sym_mat.shape

    assert gs.all(is_symmetric(sym_mat))
    sym_mat = make_symmetric(sym_mat)

    [eigenvalues, vectors] = gs.linalg.eigh(sym_mat)
    exp_eigenvalues = gs.exp(eigenvalues)

    aux = gs.einsum('ijk,ik->ijk', vectors, exp_eigenvalues)
    exp_mat = gs.einsum('ijk,ilk->ijl', aux, vectors)

    exp_mat = gs.to_ndarray(exp_mat, to_ndim=3)
    return exp_mat


def group_log(sym_mat):
    """
    Group logarithm of the Lie group of
    all invertible matrices has a straight-forward
    computation for symmetric positive definite matrices.
    """
    sym_mat = gs.to_ndarray(sym_mat, to_ndim=3)
    n_sym_mats, mat_dim, _ = sym_mat.shape

    assert gs.all(is_symmetric(sym_mat))
    sym_mat = make_symmetric(sym_mat)
    [eigenvalues, vectors] = gs.linalg.eigh(sym_mat)
    assert gs.all(eigenvalues > 0)

    log_eigenvalues = gs.log(eigenvalues)

    aux = gs.einsum('ijk,ik->ijk', vectors, log_eigenvalues)
    log_mat = gs.einsum('ijk,ilk->ijl', aux, vectors)

    log_mat = gs.to_ndarray(log_mat, to_ndim=3)
    return log_mat


class SPDMatricesSpace(EmbeddedManifold):
    """
    Class for the manifold of symmetric positive definite (SPD) matrices.
    """
    def __init__(self, n):
        assert isinstance(n, int) and n > 0
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
        mat = gs.to_ndarray(mat, to_ndim=3)
        n_mats, mat_dim, _ = mat.shape

        mask_is_symmetric = is_symmetric(mat, tolerance=tolerance)
        eigenvalues = gs.zeros((n_mats, mat_dim))
        eigenvalues[mask_is_symmetric] = gs.linalg.eigvalsh(
                                              mat[mask_is_symmetric])

        mask_pos_eigenvalues = gs.all(eigenvalues > 0)
        return mask_is_symmetric & mask_pos_eigenvalues

    def vector_from_symmetric_matrix(self, mat):
        """
        Convert the symmetric part of a symmetric matrix
        into a vector.
        """
        mat = gs.to_ndarray(mat, to_ndim=3)
        assert gs.all(is_symmetric(mat))
        mat = make_symmetric(mat)

        _, mat_dim, _ = mat.shape
        vec_dim = int(mat_dim * (mat_dim + 1) / 2)
        vec = gs.zeros(vec_dim)

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
        vec = gs.to_ndarray(vec, to_ndim=2)
        _, vec_dim = vec.shape
        mat_dim = int((gs.sqrt(8 * vec_dim + 1) - 1) / 2)
        mat = gs.zeros((mat_dim,) * 2)

        lower_triangle_indices = gs.tril_indices(mat_dim)
        diag_indices = gs.diag_indices(mat_dim)

        mat[lower_triangle_indices] = 2 * vec
        mat[diag_indices] = vec

        mat = make_symmetric(mat)
        return mat

    def random_uniform(self, n_samples=1):
        mat = 2 * gs.random.rand(n_samples, self.n, self.n) - 1

        spd_mat = group_exp(mat + gs.transpose(mat, axes=(0, 2, 1)))
        return spd_mat

    def random_tangent_vec_uniform(self, n_samples=1, base_point=None):
        if base_point is None:
            base_point = gs.eye(self.n)

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, _, _ = base_point.shape

        assert n_base_points == n_samples or n_base_points == 1

        sqrt_base_point = sqrtm(base_point)

        tangent_vec_at_id = (2 * gs.random.rand(n_samples,
                                                self.n,
                                                self.n)
                             - 1)
        tangent_vec_at_id = (tangent_vec_at_id
                             + gs.transpose(tangent_vec_at_id,
                                            axes=(0, 2, 1)))

        tangent_vec = gs.matmul(sqrt_base_point, tangent_vec_at_id)
        tangent_vec = gs.matmul(tangent_vec, sqrt_base_point)

        return tangent_vec


class SPDMetric(RiemannianMetric):

    def __init__(self, n):
        super(SPDMetric, self).__init__(dimension=int(n * (n + 1) / 2))

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """
        Compute the inner product of tangent_vec_a and tangent_vec_b
        at point base_point using the affine invariant Riemannian metric.
        """
        inv_base_point = gs.linalg.inv(base_point)

        aux_a = gs.matmul(inv_base_point, tangent_vec_a)
        aux_b = gs.matmul(inv_base_point, tangent_vec_b)
        inner_product = gs.trace(gs.matmul(aux_a, aux_b), axis1=1, axis2=2)
        inner_product = gs.to_ndarray(inner_product, to_ndim=2, axis=1)
        return inner_product

    def exp(self, tangent_vec, base_point):
        """
        Compute the Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the metric
        defined in inner_product.

        This gives a symmetric positive definite matrix.
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
        n_tangent_vecs, _, _ = tangent_vec.shape

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, mat_dim, _ = base_point.shape

        assert (n_tangent_vecs == n_base_points
                or n_tangent_vecs == 1
                or n_base_points == 1)

        sqrt_base_point = sqrtm(base_point)

        inv_sqrt_base_point = gs.linalg.inv(sqrt_base_point)

        tangent_vec_at_id = gs.matmul(inv_sqrt_base_point,
                                      tangent_vec)
        tangent_vec_at_id = gs.matmul(tangent_vec_at_id,
                                      inv_sqrt_base_point)
        exp_from_id = group_exp(tangent_vec_at_id)

        exp = gs.matmul(exp_from_id, sqrt_base_point)
        exp = gs.matmul(sqrt_base_point, exp)

        return exp

    def log(self, point, base_point):
        """
        Compute the Riemannian logarithm at point base_point,
        of point wrt the metric defined in
        inner_product.

        This gives a tangent vector at point base_point.
        """
        point = gs.to_ndarray(point, to_ndim=3)
        n_points, _, _ = point.shape

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, mat_dim, _ = base_point.shape

        assert (n_points == n_base_points
                or n_points == 1
                or n_base_points == 1)

        sqrt_base_point = gs.zeros((n_base_points,) + (mat_dim,) * 2)
        sqrt_base_point = sqrtm(base_point)

        inv_sqrt_base_point = gs.linalg.inv(sqrt_base_point)
        point_near_id = gs.matmul(inv_sqrt_base_point, point)
        point_near_id = gs.matmul(point_near_id, inv_sqrt_base_point)
        log_at_id = group_log(point_near_id)

        log = gs.matmul(sqrt_base_point, log_at_id)
        log = gs.matmul(log, sqrt_base_point)

        return log

    def geodesic(self, initial_point, initial_tangent_vec):
        return super(SPDMetric, self).geodesic(
                                      initial_point=initial_point,
                                      initial_tangent_vec=initial_tangent_vec,
                                      point_ndim=2)
