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

        self.canonical_metric = StiefelCanonicalMetric(n, p)

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


class StiefelCanonicalMetric(RiemannianMetric):

    def __init__(self, n, p):
        dimension = int(p * n - (p * (p + 1) / 2))
        super(StiefelCanonicalMetric, self).__init__(
                dimension=dimension,
                signature=(dimension, 0, 0))
        self.embedding_metric = EuclideanMetric(n*p)
        self.n = n
        self.p = p

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """
        Canonical inner product on the tangent space at base_point,
        which is different from the inner product induced by the embedding.

        Formula from:
        http://noodle.med.yale.edu/hdtag/notes/steifel_notes.pdf
        """
        tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=3)
        tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=3)
        base_point = gs.to_ndarray(base_point, to_ndim=3)
        base_point_transpose = gs.transpose(base_point, axes=(0, 2, 1))

        aux = gs.matmul(
            gs.transpose(tangent_vec_a, axes=(0, 2, 1)),
            gs.eye(self.n) - 0.5 * gs.matmul(base_point, base_point_transpose))
        inner_prod = gs.trace(gs.matmul(aux, tangent_vec_b), axis1=1, axis2=2)

        inner_prod = gs.to_ndarray(inner_prod, to_ndim=2, axis=1)
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

        matrix_a = gs.einsum(
            'nij, njk->nik',
            gs.transpose(base_point, axes=(0, 2, 1)), tangent_vec)
        matrix_k = (tangent_vec
                    - gs.einsum('nij,njk->nik', base_point, matrix_a))

        matrix_q = gs.zeros_like(matrix_k)
        matrix_r = gs.zeros(
            (matrix_k.shape[0], matrix_k.shape[2], matrix_k.shape[2]))
        for i, k in enumerate(matrix_k):
            matrix_q[i], matrix_r[i] = gs.linalg.qr(k)

        matrix_ar = gs.concatenate(
            [matrix_a,
             -gs.transpose(matrix_r, axes=(0, 2, 1))],
            axis=2)

        n_matrix_r = matrix_r.shape[0]
        if n_matrix_r == 1:
            matrix_r = gs.tile(matrix_r, (n_base_points, 1, 1))

        zeros = gs.zeros((n_base_points, p, p))
        if n_base_points == 1:
            zeros = gs.zeros((n_tangent_vecs, p, p))
        matrix_rz = gs.concatenate(
            [matrix_r,
             zeros],
            axis=2)
        block = gs.concatenate([matrix_ar, matrix_rz], axis=1)
        matrix_mn_e = gs.expm(block)

        if n_base_points == 1:
            base_point = gs.tile(base_point, (n_tangent_vecs, 1, 1))
        n_matrix_q = matrix_q.shape[0]
        if n_matrix_q == 1:
            matrix_q = gs.tile(matrix_q, (n_base_points, 1, 1))

        exp = gs.einsum(
            'nij,njk->nik',
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

        https://arxiv.org/pdf/1604.05054.pdf
        """
        # TODO(nina): Add unit tests for auxiliary functions

        def normal_component_qr(point, base_point, matrix_m):
            """
            Computes the QR decomposition of the normal component
            of a point.
            """
            matrix_k = point - gs.matmul(base_point, matrix_m)

            matrix_q = gs.zeros(matrix_k.shape)
            matrix_n = gs.zeros((n_logs, p, p))
            for i, k in enumerate(matrix_k):
                matrix_q[i], matrix_n[i] = gs.linalg.qr(k)
            return matrix_q, matrix_n

        def orthogonal_completion(matrix_m, matrix_n):
            matrix_w = gs.concatenate([matrix_m, matrix_n], axis=1)

            matrix_v = gs.zeros((
                matrix_w.shape[0],
                max(matrix_w.shape[1], matrix_w.shape[2]),
                max(matrix_w.shape[1], matrix_w.shape[2])
                ))

            for i, w in enumerate(matrix_w):
                matrix_v[i], _ = gs.linalg.qr(w, mode="complete")

            return matrix_v

        def procrustes_preprocessing(matrix_v, matrix_m, matrix_n):
            [matrix_d, matrix_s, matrix_r] = gs.linalg.svd(
                matrix_v[:, p:2*p, p:2*p])

            matrix_rd = gs.matmul(
                matrix_r, gs.transpose(matrix_d, axes=(0, 2, 1)))
            matrix_v[:, :, p:2*p] = gs.matmul(matrix_v[:, :, p:2*p], matrix_rd)
            matrix_v = gs.concatenate(
                [gs.concatenate([matrix_m, matrix_n], axis=1),
                 matrix_v[:, :, p:2*p]],
                axis=2)
            return matrix_v

        point = gs.to_ndarray(point, to_ndim=3)
        n_points, _, _ = point.shape

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, n, p = base_point.shape

        assert (n_points == n_base_points
                or n_points == 1
                or n_base_points == 1)

        if n_base_points == 1:
            base_point = gs.tile(base_point, (n_points, 1, 1))
        if n_points == 1:
            point = gs.tile(point, (n_base_points, 1, 1))
        n_logs = gs.maximum(n_base_points, n_points)

        matrix_m = gs.matmul(gs.transpose(base_point, (0, 2, 1)), point)

        matrix_q, matrix_n = normal_component_qr(point, base_point, matrix_m)

        matrix_v = orthogonal_completion(matrix_m, matrix_n)

        matrix_v = procrustes_preprocessing(matrix_v, matrix_m, matrix_n)

        for k in range(max_iter):
            matrix_lv = gs.logm(matrix_v)

            matrix_c = matrix_lv[:, p:2*p, p:2*p]
            norm_matrix_c = gs.linalg.norm(matrix_c, ord=2, axis=(1, 2))

            if gs.all(norm_matrix_c < tol):
                # Convergence achieved
                break

            matrix_phi = gs.expm(-matrix_c)
            matrix_v[:, :, p:2*p] = gs.matmul(
                matrix_v[:, :, p:2*p], matrix_phi)

        matrix_xv = gs.matmul(base_point, matrix_lv[:, 0:p, 0:p])
        matrix_qv = gs.matmul(matrix_q, matrix_lv[:, p:2*p, 0:p])

        return matrix_xv + matrix_qv

    def retraction(self, tangent_vec, base_point):
        """
        Retraction map, based on QR-decomposion:
        P_x(V) = qf(X + V)
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
        n_tangent_vecs, _, _ = tangent_vec.shape

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, n, p = base_point.shape

        assert (n_tangent_vecs == n_base_points
                or n_tangent_vecs == 1
                or n_base_points == 1)

        if n_base_points == 1:
            base_point = gs.tile(base_point, (n_tangent_vecs, 1, 1))
        if n_tangent_vecs == 1:
            tangent_vec = gs.tile(tangent_vec, (n_base_points, 1, 1))
        n_retractions = gs.maximum(n_base_points, n_tangent_vecs)

        # TODO(nina): remove for loop, when qr is vectorized
        matrix_q = gs.zeros_like(base_point)
        matrix_r = gs.zeros((n_retractions, p, p))

        for i, k in enumerate(base_point + tangent_vec):
            matrix_q[i], matrix_r[i] = gs.linalg.qr(k)

        # flipping signs
        # TODO(nina): remove for loop, when diag and diagonal are vectorized
        result = gs.zeros_like(base_point)
        for i, _ in enumerate(matrix_r):
            result[i] = gs.matmul(
                matrix_q[i],
                gs.diag(gs.sign(gs.sign(gs.diagonal(matrix_r[i])) + 0.5)))

        return result

    def lifting(self, point, base_point):
        """
        Lifting map, based on QR-decomposion:
        P_x^{-1}(Q) = QR - X
        """
        point = gs.to_ndarray(point, to_ndim=3)
        n_points, _, _ = point.shape

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, p, n = base_point.shape

        assert (n_points == n_base_points
                or n_points == 1
                or n_base_points == 1)

        if n_base_points == 1:
            base_point = gs.tile(base_point, (n_points, 1, 1))
        if n_points == 1:
            point = gs.tile(point, (n_base_points, 1, 1))
        n_liftings = gs.maximum(n_base_points, n_points)

        def make_minor(i, matrix):
            return matrix[:i+1, :i+1]

        def make_column_r(i, matrix):
            if i == 0:
                assert matrix[0, 0] > 0, 'M[0,0] <= 0'
                return gs.array([1. / matrix[0, 0]])
            else:
                # get principal minor
                matrix_m_i = make_minor(i, matrix_m_k)

                assert gs.linalg.det(matrix_m_i) != 0
                inv_matrix_m_i = gs.linalg.inv(matrix_m_i)

                b_i = make_b(i, matrix_m_k, columns_list)

                column_r_i = gs.matmul(inv_matrix_m_i, b_i)

                assert column_r_i[i] > 0, '(r_i)_i <= 0'
                return column_r_i

        def make_b(i, matrix, list_matrices_r):
            b = gs.ones(i+1)

            for j in range(i):
                b[j] = - gs.matmul(
                    matrix[i, :j+1], list_matrices_r[j])

            return b

        matrix_r = gs.zeros((n_liftings, n, n))
        matrix_m = gs.matmul(gs.transpose(base_point, axes=(0, 2, 1)), point)

        for k in range(n_liftings):
            columns_list = []
            matrix_m_k = matrix_m[k]

            for i in range(n):
                column_r_i = make_column_r(i, matrix_m_k)
                columns_list.append(column_r_i)
                matrix_r[k, :len(column_r_i), i] = gs.array(column_r_i)

        return gs.matmul(point, matrix_r) - base_point
