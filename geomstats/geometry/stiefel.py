"""Stiefel manifold St(n,p).

A set of all orthonormal p-frames in n-dimensional space, where p <= n
"""

import geomstats.backend as gs
import geomstats.errors
import geomstats.vectorization
from geomstats import algebra_utils
from geomstats.geometry.base import EmbeddedManifold
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.symmetric_matrices import SymmetricMatrices

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
        matrices = Matrices(n, p)
        super(Stiefel, self).__init__(
            dim=dim, embedding_space=matrices,
            submersion=lambda x: matrices.mul(matrices.transpose(x), x),
            value=gs.eye(p),
            tangent_submersion=lambda v, x: 2 * matrices.to_symmetric(
                matrices.mul(matrices.transpose(x), v)),
            metric=StiefelCanonicalMetric(n, p))

        self.n = n
        self.p = p
        self.canonical_metric = self.metric

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
        aux = Matrices.mul(std_normal_transpose, std_normal)
        inv_sqrt_aux = SymmetricMatrices.powerm(aux, -1. / 2)
        samples = Matrices.mul(std_normal, inv_sqrt_aux)

        return samples

    def random_point(self, n_samples=1, bound=1.):
        r"""Sample on St(n,p) from the uniform distribution.

        If :math:`Z(p,n) \sim N(0,1)`, then :math:`St(n,p) \sim U`,
        according to Haar measure:
        :math:`St(n,p) := Z(Z^TZ)^{-1/2}`.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Unused here.

        Returns
        -------
        samples : array-like, shape=[..., n, p]
            Samples on the Stiefel manifold.
        """
        return self.random_uniform(n_samples)

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        Inspired by the method of Pymanopt.

        Parameters
        ----------
        vector : array-like, shape=[..., n, p]
            Vector.
        base_point : array-like, shape=[..., n, p]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n, p]
            Tangent vector at base point.
        """
        aux = Matrices.mul(Matrices.transpose(base_point), vector)
        sym_aux = Matrices.to_symmetric(aux)
        return vector - Matrices.mul(base_point, sym_aux)

    def projection(self, point):
        """Project a close enough matrix to the Stiefel manifold.

        A singular value decomposition is used, and all singular values are
        set to 1 [Absil]_

        Parameters
        ----------
        point : array-like, shape=[..., n, p]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., n, p]

        References
        ----------
        ..[Absil]   Absil, Pierre-Antoine, and Jérôme Malick.
                    “Projection-like Retractions on Matrix Manifolds.”
                    SIAM Journal on Optimization 22, no. 1 (January 2012):
                     135–58. https://doi.org/10.1137/100802529.
        """
        mat_u, _, mat_v = gs.linalg.svd(point)
        return Matrices.mul(mat_u[..., :, :self.p], mat_v)


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
        base_point_transpose = Matrices.transpose(base_point)

        aux = gs.matmul(
            Matrices.transpose(tangent_vec_a),
            gs.eye(self.n) - 0.5 * gs.matmul(base_point, base_point_transpose))
        inner_prod = Matrices.trace_product(aux, tangent_vec_b)

        return inner_prod

    def exp(self, tangent_vec, base_point, **kwargs):
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
        p = self.p
        matrix_a = Matrices.mul(Matrices.transpose(base_point), tangent_vec)
        matrix_k = (tangent_vec - Matrices.mul(base_point, matrix_a))

        matrix_q, matrix_r = gs.linalg.qr(matrix_k)

        matrix_ar = gs.concatenate(
            [matrix_a, -Matrices.transpose(matrix_r)], axis=-1)

        zeros = gs.zeros_like(tangent_vec)[..., :p, :p]
        matrix_rz = gs.concatenate([matrix_r, zeros], axis=-1)
        block = gs.concatenate([matrix_ar, matrix_rz], axis=-2)
        matrix_mn_e = gs.linalg.expm(block)

        exp = (
            Matrices.mul(base_point, matrix_mn_e[..., :p, :p])
            + Matrices.mul(matrix_q, matrix_mn_e[..., p:, :p]))
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
        matrix_w = gs.concatenate([matrix_m, matrix_n], axis=-2)
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
        [matrix_d, _, matrix_r] = gs.linalg.svd(matrix_v[..., p:, p:])
        j_matrix = gs.eye(p)
        for i in range(1, p):
            matrix_rd = Matrices.mul(
                matrix_r, j_matrix, Matrices.transpose(matrix_d))
            sub_matrix_v = gs.matmul(matrix_v[..., :, p:], matrix_rd)
            matrix_v = gs.concatenate([
                gs.concatenate([matrix_m, matrix_n], axis=-2),
                sub_matrix_v], axis=-1)
            det = gs.linalg.det(matrix_v)
            if gs.all(det > 0):
                break
            ones = gs.ones(p)
            reflection_vec = gs.concatenate(
                [ones[:-i], gs.array([-1.] * i)], axis=0)
            mask = gs.cast(det < 0, gs.float32)
            sign = (mask[..., None] * reflection_vec
                    + (1. - mask)[..., None] * ones)
            j_matrix = algebra_utils.from_vector_to_diagonal_matrix(sign)
        return matrix_v

    def log(self, point, base_point, max_iter=30, tol=gs.atol, **kwargs):
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
        p = self.p

        transpose_base_point = Matrices.transpose(base_point)
        matrix_m = gs.matmul(transpose_base_point, point)

        matrix_q, matrix_n = self._normal_component_qr(
            point, base_point, matrix_m)

        matrix_v = self._orthogonal_completion(matrix_m, matrix_n)
        matrix_v = self._procrustes_preprocessing(
            p, matrix_v, matrix_m, matrix_n)
        matrix_v = gs.to_ndarray(matrix_v, to_ndim=3)
        result = []
        for x in matrix_v:
            result.append(self._iter_log(p, x, max_iter, tol))
        result = gs.stack(result)
        matrix_lv = (
            result[0] if (point.ndim == 2) and (base_point.ndim == 2)
            else result)
        matrix_xv = gs.matmul(base_point, matrix_lv[..., :p, :p])
        matrix_qv = gs.matmul(matrix_q, matrix_lv[..., p:, :p])

        return matrix_xv + matrix_qv

    @staticmethod
    def _iter_log(p, matrix_v, max_iter, tol):
        matrix_lv = gs.zeros_like(matrix_v)
        for _ in range(max_iter):
            matrix_lv = gs.linalg.logm(matrix_v)
            matrix_c = matrix_lv[..., p:, p:]
            norm_matrix_c = gs.linalg.norm(matrix_c)
            if norm_matrix_c <= tol:
                break

            matrix_phi = gs.linalg.expm(-Matrices.to_skew_symmetric(matrix_c))
            aux_matrix = gs.matmul(
                matrix_v[..., :, p:], matrix_phi)
            matrix_v = gs.concatenate(
                [matrix_v[..., :, :p], aux_matrix], axis=-1)
        return matrix_lv

    @staticmethod
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
        matrix_q, matrix_r = gs.linalg.qr(base_point + tangent_vec)

        diagonal = gs.diagonal(matrix_r, axis1=-2, axis2=-1)
        sign = gs.sign(gs.sign(diagonal) + 0.5)
        diag = algebra_utils.from_vector_to_diagonal_matrix(sign)
        result = Matrices.mul(matrix_q, diag)

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
        matrix_m = gs.matmul(Matrices.transpose(base_point), point)

        for k in range(n_liftings):
            columns_list = []
            matrix_m_k = matrix_m[k]

            for j in range(n):
                column_r_j = _make_column_r(j, matrix_m_k)
                columns_list.append(column_r_j)
                matrix_r[k, :len(column_r_j), j] = gs.array(column_r_j)

        return gs.matmul(point, matrix_r) - base_point
