"""
Stiefel manifold St(n,p),
a set of all orthonormal p-frames in n-dimensional space,
where p <= n
"""

import geomstats.backend as gs

from geomstats.embedded_manifold import EmbeddedManifold
from geomstats.euclidean_space import EuclideanMetric
from geomstats.matrices_space import MatricesSpace
from geomstats.riemannian_metric import RiemannianMetric

TOLERANCE = 1e-6
EPSILON = 1e-6


class Stiefel(EmbeddedManifold):
    """
    Class for Stiefel manifolds St(n,p),
    a set of all orthonormal p-frames in n-dimensional space,
    where p <= n.
    """
    def __init__(self, n, p):
        assert isinstance(n, int) and isinstance(p, int)
        assert p <= n

        self.n = n
        self.p = p

        dimension = int(p * n - (p * (p + 1) / 2))
        super(Stiefel, self).__init__(
              dimension=dimension,
              embedding_manifold=MatricesSpace(n, p))

        self.euclidean_metric = StiefelEuclideanMetric(n, p)

    def belongs(self, point, tolerance=TOLERANCE):
        """
        Evaluate if a point belongs to St(n,p),
        i.e. if it is a p-frame in n-dimensional space,
        and it is orthonormal.
        """
        point = gs.to_ndarray(point, to_ndim=3)
        n_points, n, p = point.shape

        if (n, p) != (self.n, self.p):
            return gs.array([[False]] * n_points)

        point_transpose = gs.transpose(point, axes=(0, 2, 1))
        diff = gs.matmul(point_transpose, point) - gs.eye(p)

        diff_norm = gs.norm(diff, axis=(1, 2))
        belongs = gs.less_equal(diff_norm, tolerance)

        belongs = gs.to_ndarray(belongs, to_ndim=2)
        return belongs

    def random_uniform(self, n_samples=1):
        """
        Sample on St(n,p) with the uniform distribution.

        If Z(p,n) ~ N(0,1), then St(n,p) ~ U, according to Haar measure:
        St(n,p) := Z(Z^TZ)^{-1/2}
        """
        std_normal = gs.random.normal(size=(n_samples, self.n, self.p))
        std_normal_transpose = gs.transpose(std_normal, axes=(0, 2, 1))
        aux = gs.matmul(std_normal_transpose, std_normal)
        sqrt_aux = gs.sqrtm(aux)
        inv_sqrt_aux = gs.linalg.inv(sqrt_aux)

        return gs.matmul(std_normal, inv_sqrt_aux)


class StiefelEuclideanMetric(RiemannianMetric):

    def __init__(self, n, p):
        dimension = int(p * n - (p * (p + 1) / 2))
        super(StiefelEuclideanMetric, self).__init__(
                dimension=dimension,
                signature=(dimension, 0, 0))
        self.embedding_metric = EuclideanMetric(n*p)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """
        Compute the Frobenius inner product of tangent_vec_a and tangent_vec_b
        at base_point using the metric of the embedding space.
        """
        tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=3)
        n_tangent_vecs_a, _, _ = tangent_vec_a.shape

        tangent_vec_b = gs.to_ndarray(tangent_vec_a, to_ndim=3)
        n_tangent_vecs_b, _, _ = tangent_vec_b.shape

        assert n_tangent_vecs_a == n_tangent_vecs_b

        inner_prod = gs.einsum("nij,nij->n", tangent_vec_a, tangent_vec_b)

        return inner_prod

    def exp(self, tangent_vec, base_point):
        """
        Riemannian exponential of a tangent vector wrt to a base point.
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
        n_tangent_vecs, _, _ = tangent_vec.shape

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, n, p = base_point.shape

        assert (n_tangent_vecs == n_base_points
                or n_tangent_vecs == 1
                or n_base_points == 1)

        matrix_a = gs.matmul(
            gs.transpose(base_point, axes=(0, 2, 1)), tangent_vec)
        matrix_k = tangent_vec - gs.matmul(base_point, matrix_a)

        matrix_q = gs.zeros(matrix_k.shape)
        matrix_r = gs.zeros(
            (matrix_k.shape[0], matrix_k.shape[2], matrix_k.shape[2]))
        for i, k in enumerate(matrix_k):
            matrix_q[i], matrix_r[i] = gs.linalg.qr(k)

        matrix_ar = gs.concatenate(
            [matrix_a,
             -gs.transpose(matrix_r, axes=(0, 2, 1))],
            axis=2)
        matrix_rz = gs.concatenate(
            [matrix_r,
             gs.zeros((n_base_points, p, p))],
            axis=2)
        block = gs.concatenate([matrix_ar, matrix_rz], axis=1)
        matrix_mn_e = gs.expm(block)

        exp = gs.matmul(
            gs.concatenate(
                [base_point,
                 matrix_q],
                axis=2),
            matrix_mn_e[:, :, 0:p])

        return exp

    def log(self, point, base_point, max_iter=100, tol=1e-6):
        """
        Riemannian logarithm of a point wrt a base point.

        Based on:
        Zimmermann, Ralf
        "A Matrix-Algebraic Algorithm for the Riemannian Logarithm
        on the Stiefel Manifold under the Canonical Metric"
        SIAM J. Matrix Anal. & Appl., 38(2), 322–342.
        """
        point = gs.to_ndarray(point, to_ndim=3)
        n_points, _, _ = point.shape

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, n, p = base_point.shape

        assert (n_points == n_base_points
                or n_points == 1
                or n_base_points == 1)

        matrix_m = gs.matmul(gs.transpose(base_point, (0, 2, 1)), point)

        # QR of normal component of a point
        matrix_k = point - gs.matmul(base_point, matrix_m)

        matrix_q = gs.zeros(matrix_k.shape)
        matrix_n = gs.zeros(
            (matrix_k.shape[0], matrix_k.shape[2], matrix_k.shape[2]))
        for i, k in enumerate(matrix_k):
            matrix_q[i], matrix_n[i] = gs.linalg.qr(k)

        # orthogonal completion
        matrix_w = gs.concatenate([matrix_m, matrix_n], axis=1)

        matrix_v = gs.zeros((
            matrix_w.shape[0],
            max(matrix_w.shape[1], matrix_w.shape[2]),
            max(matrix_w.shape[1], matrix_w.shape[2])
            ))

        for i, w in enumerate(matrix_w):
            matrix_v[i], _ = gs.linalg.qr(w, mode="complete")

        # Procrustes preprocessing
        [matrix_d, matrix_s, matrix_r] = gs.linalg.svd(
            matrix_v[:, p:2*p, p:2*p])

        matrix_rd = gs.matmul(matrix_r, gs.transpose(matrix_d, axes=(0, 2, 1)))
        matrix_v[:, :, p:2*p] = gs.matmul(matrix_v[:, :, p:2*p], matrix_rd)
        matrix_v = gs.concatenate(
            [gs.concatenate([matrix_m, matrix_n], axis=1),
             matrix_v[:, :, p:2*p]],
            axis=2)

        for k in range(max_iter):

            matrix_lv = gs.logm(matrix_v)

            matrix_c = matrix_lv[:, p:2*p, p:2*p]
            norm_matrix_c = gs.linalg.norm(matrix_c, ord=2, axis=(1, 2))

            if norm_matrix_c < tol:
                # print("Converged in {} iterations".format(k+1))
                break

            matrix_phi = gs.expm(-matrix_c)
            matrix_v[:, :, p:2*p] = gs.matmul(
                matrix_v[:, :, p:2*p], matrix_phi)

        matrix_xv = gs.matmul(base_point, matrix_lv[:, 0:p, 0:p])
        matrix_qv = gs.matmul(matrix_q, matrix_lv[:, p:2*p, 0:p])

        return matrix_xv + matrix_qv
