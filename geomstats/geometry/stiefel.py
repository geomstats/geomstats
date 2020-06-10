"""Stiefel manifold St(n,p).

A set of all orthonormal p-frames in n-dimensional space, where p <= n
"""

import geomstats.backend as gs
import geomstats.errors
import geomstats.vectorization
from geomstats import algebra_utils
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.riemannian_metric import RiemannianMetric

TOLERANCE = 1e-5
EPSILON = 1e-6


class Stiefel(EmbeddedManifold):
    """Class for Stiefel manifolds St(n,p).

    A set of all orthonormal p-frames in n-dimensional space,
    where p <= n.

    Parameters
    ----------
    n : int
        Dimension of the ambient vector space.
    p : int
        Number of basis vectors in the orthonormal frame.
    """

    def __init__(self, n, p):
        geomstats.errors.check_integer(n, 'n')
        geomstats.errors.check_integer(p, 'p')
        if p > n:
            raise ValueError('p needs to be smaller than n.')

        dim = int(p * n - (p * (p + 1) / 2))
        super(Stiefel, self).__init__(
            dim=dim,
            embedding_manifold=Matrices(n, p))

        self.n = n
        self.p = p
        self.canonical_metric = StiefelCanonicalMetric(n, p)

    @geomstats.vectorization.decorator(['else', 'matrix', 'else'])
    def belongs(self, point, tolerance=TOLERANCE):
        """Test if a point belongs to St(n,p).

        Test whether the point is a p-frame in n-dimensional space,
        and it is orthonormal.

        Parameters
        ----------
        point : array-like, shape=[..., n, p]
            Point.
        tolerance : float, optional
            Tolerance at which to evaluate.
            Optional, default: 1e-5.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Array of booleans evaluating if the corresponding points
            belong to the Stiefel manifold.
        """
        n_points, n, p = point.shape

        if (n, p) != (self.n, self.p):
            return gs.array([False] * n_points)

        point_transpose = Matrices.transpose(point)
        identity = gs.eye(p)
        diff = gs.einsum(
            '...ij,...jk->...ik', point_transpose, point) - identity

        diff_norm = gs.linalg.norm(diff, axis=(-2, -1))
        belongs = gs.less_equal(diff_norm, tolerance)
        belongs = gs.to_ndarray(belongs, to_ndim=1)
        return belongs

    @staticmethod
    def to_grassmannian(point):
        r"""Project a point of St(n, p) to Gr(n, p).

        If :math:`U \in St(n, p)` is an orthonormal frame,
        return the orthogonal projector :math:`P = U U^T`
        onto the subspace of :math:`\mathbb{R}^n` spanned by
        :math:`U`.

        Parameters
        ----------
        point : array-like, shape=[..., n, p]
            Point.

        Returns
        -------
        projected : array-like, shape=[..., n, n]
            Projected point.
        """
        return Matrices.mul(point, Matrices.transpose(point))

    def random_uniform(self, n_samples=1):
        r"""Sample on St(n,p) from the uniform distribution.

        If :math:`Z(p,n) \sim N(0,1)`, then :math:`St(n,p) \sim U`,
        according to Haar measure:
        :math:`St(n,p) := Z(Z^TZ)^{-1/2}`.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n, p]
            Samples on the Stiefel manifold.
        """
        n, p = self.n, self.p
        size = (n_samples, n, p) if n_samples != 1 else (n, p)

        std_normal = gs.random.normal(size=size)
        std_normal_transpose = Matrices.transpose(std_normal)
        aux = gs.einsum(
            '...ij,...jk->...ik', std_normal_transpose, std_normal)
        sqrt_aux = gs.linalg.sqrtm(aux)
        inv_sqrt_aux = gs.linalg.inv(sqrt_aux)
        samples = gs.einsum(
            '...ij,...jk->...ik', std_normal, inv_sqrt_aux)

        return samples


class StiefelCanonicalMetric(RiemannianMetric):
    """Class that defines the canonical metric for Stiefel manifolds.

    Parameters
    ----------
    n : int
        Dimension of the ambient vector space.
    p : int
        Number of basis vectors in the orthonormal frames.
    """

    def __init__(self, n, p):
        dim = int(p * n - (p * (p + 1) / 2))
        super(StiefelCanonicalMetric, self).__init__(
            dim=dim,
            signature=(dim, 0, 0))
        self.embedding_metric = EuclideanMetric(n * p)
        self.n = n
        self.p = p

    @geomstats.vectorization.decorator(['else', 'matrix', 'matrix', 'matrix'])
    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        r"""Compute the inner-product of two tangent vectors at a base point.

        Canonical inner-product on the tangent space at `base_point`,
        which is different from the inner-product induced by the embedding
        (see [RLSMRZ2017]_).

        .. math::

            \langle\Delta, \tilde{\Delta}\rangle_{U}=\operatorname{tr}
            \left(\Delta^{T}\left(I-\frac{1}{2} U U^{T}\right)
            \tilde{\Delta}\right)

        References
        ----------
        .. [RLSMRZ2017] R Zimmermann. A matrix-algebraic algorithm for the
          Riemannian logarithm on the Stiefel manifold under the canonical
          metric. SIAM Journal on Matrix Analysis and Applications 38 (2),
          322-342, 2017. https://epubs.siam.org/doi/pdf/10.1137/16M1074485

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, p]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., n, p]
            Second tangent vector at base point.
        base_point : array-like, shape=[..., n, p]
            Point in the Stiefel manifold.

        Returns
        -------
        inner_prod : array-like, shape=[..., 1]
            Inner-product of the two tangent vectors.
        """
        base_point_transpose = gs.transpose(base_point, axes=(0, 2, 1))

        aux = gs.matmul(
            gs.transpose(tangent_vec_a, axes=(0, 2, 1)),
            gs.eye(self.n) - 0.5 * gs.matmul(base_point, base_point_transpose))
        inner_prod = gs.trace(gs.matmul(aux, tangent_vec_b), axis1=1, axis2=2)

        inner_prod = gs.to_ndarray(inner_prod, to_ndim=2, axis=1)
        return inner_prod

    @geomstats.vectorization.decorator(['else', 'matrix', 'matrix'])
    def exp(self, tangent_vec, base_point):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, p]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., n, p]
            Point in the Stiefel manifold.

        Returns
        -------
        exp : array-like, shape=[..., n, p]
            Point in the Stiefel manifold equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        n_tangent_vecs, _, _ = tangent_vec.shape
        n_base_points, _, p = base_point.shape

        if not (n_tangent_vecs == n_base_points
                or n_tangent_vecs == 1
                or n_base_points == 1):
            raise NotImplementedError

        if n_tangent_vecs == 1:
            tangent_vec = gs.tile(tangent_vec, (n_base_points, 1, 1))

        if n_base_points == 1:
            base_point = gs.tile(base_point, (n_tangent_vecs, 1, 1))

        matrix_a = gs.einsum(
            'nij, njk->nik',
            gs.transpose(base_point, axes=(0, 2, 1)), tangent_vec)
        matrix_k = (tangent_vec
                    - gs.einsum('nij,njk->nik', base_point, matrix_a))

        matrix_q, matrix_r = gs.linalg.qr(matrix_k)

        matrix_ar = gs.concatenate(
            [matrix_a,
             -gs.transpose(matrix_r, axes=(0, 2, 1))],
            axis=2)

        zeros = gs.zeros(
            (gs.maximum(n_base_points, n_tangent_vecs), p, p))

        matrix_rz = gs.concatenate(
            [matrix_r,
             zeros],
            axis=2)
        block = gs.concatenate([matrix_ar, matrix_rz], axis=1)
        matrix_mn_e = gs.linalg.expm(block)

        exp = gs.einsum(
            'nij,njk->nik',
            gs.concatenate(
                [base_point,
                 matrix_q],
                axis=2),
            matrix_mn_e[:, :, 0:p])

        return exp

    @staticmethod
    def _normal_component_qr(point, base_point, matrix_m):
        """Compute the QR decomposition of the normal component of a point.

        Parameters
        ----------
        point : array-like, shape=[..., n, p]
        base_point : array-like, shape=[..., n, p]
        matrix_m : array-like

        Returns
        -------
        matrix_q : array-like
        matrix_n : array-like
        """
        matrix_k = point - gs.matmul(base_point, matrix_m)
        matrix_q, matrix_n = gs.linalg.qr(matrix_k)
        return matrix_q, matrix_n

    @staticmethod
    def _orthogonal_completion(matrix_m, matrix_n):
        """Orthogonal matrix completion.

        Parameters
        ----------
        matrix_m : array-like
        matrix_n : array-like

        Returns
        -------
        matrix_v : array-like
        """
        matrix_w = gs.concatenate([matrix_m, matrix_n], axis=1)

        matrix_v, _ = gs.linalg.qr(matrix_w, mode='complete')

        return matrix_v

    @staticmethod
    def _procrustes_preprocessing(p, matrix_v, matrix_m, matrix_n):
        """Procrustes preprocessing.

        Parameters
        ----------
        matrix_v : array-like
        matrix_m : array-like
        matrix_n : array-like

        Returns
        -------
        matrix_v : array-like
        """
        [matrix_d, _, matrix_r] = gs.linalg.svd(
            matrix_v[:, p:2 * p, p:2 * p])

        matrix_rd = gs.matmul(
            matrix_r, gs.transpose(matrix_d, axes=(0, 2, 1)))
        sub_matrix_v = gs.matmul(matrix_v[:, :, p:2 * p], matrix_rd)
        matrix_v = gs.concatenate(
            [gs.concatenate([matrix_m, matrix_n], axis=1),
             sub_matrix_v],
            axis=2)
        return matrix_v

    @geomstats.vectorization.decorator(['else', 'matrix', 'matrix', 'else'])
    def log(self, point, base_point, max_iter=30, tol=1e-6):
        """Compute the Riemannian logarithm of a point.

        Based on [ZR2017]_.

        References
        ----------
        .. [ZR2017] Zimmermann, Ralf. "A Matrix-Algebraic Algorithm for the
          Riemannian Logarithm on the Stiefel Manifold under the Canonical
          Metric" SIAM J. Matrix Anal. & Appl., 38(2), 322–342, 2017.
          https://arxiv.org/pdf/1604.05054.pdf

        Parameters
        ----------
        point : array-like, shape=[..., n, p]
            Point in the Stiefel manifold.
        base_point : array-like, shape=[..., n, p]
            Point in the Stiefel manifold.
        max_iter: int
            Maximum number of iterations to perform during the algorithm.
            Optional, default: 30.
        tol: float
            Tolerance to reach convergence. The matrix 2-norm is used as
            criterion.
            Optional, default: 1e-6.

        Returns
        -------
        log : array-like, shape=[..., dim + 1]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        p = base_point.shape[-1]

        transpose_base_point = Matrices.transpose(base_point)
        matrix_m = gs.matmul(transpose_base_point, point)

        matrix_q, matrix_n = StiefelCanonicalMetric._normal_component_qr(
            point, base_point, matrix_m)

        matrix_v = StiefelCanonicalMetric._orthogonal_completion(
            matrix_m, matrix_n)

        matrix_v = StiefelCanonicalMetric._procrustes_preprocessing(
            p, matrix_v, matrix_m, matrix_n)

        for _ in range(max_iter):
            matrix_lv = gs.linalg.logm(matrix_v)

            matrix_c = matrix_lv[:, p:2 * p, p:2 * p]

            norm_matrix_c = gs.linalg.norm(matrix_c)

            if gs.less_equal(norm_matrix_c, tol):
                break

            matrix_phi = gs.linalg.expm(-matrix_c)

            aux_matrix = gs.matmul(
                matrix_v[:, :, p:2 * p], matrix_phi)

            matrix_v = gs.concatenate(
                [matrix_v[:, :, 0:p],
                 aux_matrix],
                axis=2)

        matrix_xv = gs.matmul(base_point, matrix_lv[:, 0:p, 0:p])
        matrix_qv = gs.matmul(matrix_q, matrix_lv[:, p:2 * p, 0:p])

        return matrix_xv + matrix_qv

    @staticmethod
    @geomstats.vectorization.decorator(['matrix', 'matrix'])
    def retraction(tangent_vec, base_point):
        """Compute the retraction of a tangent vector.

        This computation is based on the QR-decomposition.

        e.g. :math:`P_x(V) = qf(X + V)`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, p]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., n, p]
            Point in the Stiefel manifold.

        Returns
        -------
        exp : array-like, shape=[..., n, p]
            Point in the Stiefel manifold equal to the retraction
            of tangent_vec at the base point.
        """
        n_tangent_vecs, _, _ = tangent_vec.shape
        n_base_points, _, _ = base_point.shape

        if not (n_tangent_vecs == n_base_points
                or n_tangent_vecs == 1
                or n_base_points == 1):
            raise NotImplementedError

        matrix_q, matrix_r = gs.linalg.qr(base_point + tangent_vec)

        diagonal = gs.diagonal(matrix_r, axis1=1, axis2=2)
        sign = gs.sign(gs.sign(diagonal) + 0.5)
        diag = algebra_utils.from_vector_to_diagonal_matrix(sign)
        result = gs.einsum('nij,njk->nik', matrix_q, diag)

        return result

    @staticmethod
    @geomstats.vectorization.decorator(['matrix', 'matrix'])
    def lifting(point, base_point):
        """Compute the lifting of a point.

        This computation is based on the QR-decomposion.

        e.g. :math:`P_x^{-1}(Q) = QR - X`.

        Parameters
        ----------
        point : array-like, shape=[..., n, p]
            Point in the Stiefel manifold.
        base_point : array-like, shape=[..., n, p]
            Point in the Stiefel manifold.

        Returns
        -------
        log : array-like, shape=[..., dim + 1]
            Tangent vector at the base point equal to the lifting
            of point at the base point.
        """
        n_points, _, _ = point.shape
        n_base_points, _, n = base_point.shape

        if not (n_points == n_base_points
                or n_points == 1
                or n_base_points == 1):
            raise NotImplementedError

        n_liftings = gs.maximum(n_base_points, n_points)

        def _make_minor(i, matrix):
            return matrix[:i + 1, :i + 1]

        def _make_column_r(i, matrix):
            if i == 0:
                if matrix[0, 0] <= 0:
                    raise ValueError('M[0,0] <= 0')
                return gs.array([1. / matrix[0, 0]])
            matrix_m_i = _make_minor(i, matrix_m_k)
            inv_matrix_m_i = gs.linalg.inv(matrix_m_i)
            b_i = _make_b(i, matrix_m_k, columns_list)
            column_r_i = gs.matmul(inv_matrix_m_i, b_i)

            if column_r_i[i] <= 0:
                raise ValueError('(r_i)_i <= 0')
            return column_r_i

        def _make_b(i, matrix, list_matrices_r):
            b = gs.ones(i + 1)

            for j in range(i):
                b[j] = - gs.matmul(
                    matrix[i, :j + 1], list_matrices_r[j])

            return b

        matrix_r = gs.zeros((n_liftings, n, n))
        matrix_m = gs.matmul(gs.transpose(base_point, axes=(0, 2, 1)), point)

        for k in range(n_liftings):
            columns_list = []
            matrix_m_k = matrix_m[k]

            for i in range(n):
                column_r_i = _make_column_r(i, matrix_m_k)
                columns_list.append(column_r_i)
                matrix_r[k, :len(column_r_i), i] = gs.array(column_r_i)

        return gs.matmul(point, matrix_r) - base_point
