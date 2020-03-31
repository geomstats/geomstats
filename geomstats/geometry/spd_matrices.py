"""The manifold of symmetric positive definite (SPD) matrices."""

import math

import geomstats.backend as gs
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.riemannian_metric import RiemannianMetric

EPSILON = 1e-6
TOLERANCE = 1e-12


class SPDMatrices(EmbeddedManifold):
    """Class for the manifold of symmetric positive definite (SPD) matrices."""

    def __init__(self, n):
        assert isinstance(n, int) and n > 0
        super(SPDMatrices, self).__init__(
            dimension=int(n * (n + 1) / 2),
            embedding_manifold=GeneralLinear(n=n))
        self.n = n

    @staticmethod
    def belongs(mat, atol=TOLERANCE):
        """Check if a matrix is symmetric and invertible."""
        # TODO (opeltre): check positivity, implying invertibility.
        #
        # note : vectorized "and" on numpy works with:
        #       [bool] * [bool] -> bool
        # but does not on tf.
        return Matrices.is_symmetric(mat)

    def vector_from_symmetric_matrix(self, mat):
        """Convert the symmetric part of a symmetric matrix into a vector."""
        mat = gs.to_ndarray(mat, to_ndim=3)
        assert gs.all(self.embedding_manifold.is_symmetric(mat))
        mat = self.embedding_manifold.make_symmetric(mat)

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
        """Convert a vector into a symmetric matrix."""
        vec = gs.to_ndarray(vec, to_ndim=2)
        _, vec_dim = vec.shape
        mat_dim = int((gs.sqrt(8 * vec_dim + 1) - 1) / 2)
        mat = gs.zeros((mat_dim,) * 2)

        lower_triangle_indices = gs.tril_indices(mat_dim)
        diag_indices = gs.diag_indices(mat_dim)

        mat[lower_triangle_indices] = 2 * vec
        mat[diag_indices] = vec

        mat = self.embedding_manifold.make_symmetric(mat)
        return mat

    def random_uniform(self, n_samples=1):
        """Define a log-uniform random sample of SPD matrices."""
        mat = 2 * gs.random.rand(n_samples, self.n, self.n) - 1
        spd_mat = GeneralLinear.exp(mat + Matrices.transpose(mat))

        return spd_mat

    def random_tangent_vec_uniform(self, n_samples=1, base_point=None):
        """Define a uniform random sample of tangent vectors."""
        if base_point is None:
            base_point = gs.eye(self.n)

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, _, _ = base_point.shape

        assert n_base_points == n_samples or n_base_points == 1
        if n_base_points == 1:
            base_point = gs.tile(base_point, (n_samples, 1, 1))

        sqrt_base_point = gs.linalg.sqrtm(base_point)

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

    def aux_differential_power(self, power, tangent_vec, base_point):
        """Compute the differential of the matrix power.

        Auxiliary function to the functions differential_power and
        inverse_differential_power.

        Parameters
        ----------
        power : int
        tangent_vec : array_like, shape=[n_samples, n, n]
            Tangent vectors.
        base_point : array_like, shape=[n_samples, n, n]
            Base points.

        Returns
        -------
        eigvectors : array-like, shape=[n_samples, n, n]
        transp_eigvectors : array-like, shape=[n_samples, n, n]
        numerator : array-like, shape=[n_samples, n, n]
        denominator : array-like, shape=[n_samples, n, n]
        temp_result : array-like, shape=[n_samples, n, n]
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
        n_tangent_vecs, _, _ = tangent_vec.shape
        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, _, _ = base_point.shape

        assert (n_tangent_vecs == n_base_points
                or n_base_points == 1
                or n_tangent_vecs == 1)

        eigvalues, eigvectors = gs.linalg.eigh(base_point)
        eigvalues = gs.to_ndarray(eigvalues, to_ndim=3, axis=1)
        transp_eigvalues = gs.transpose(eigvalues, (0, 2, 1))

        if power == 0:
            powered_eigvalues = gs.log(eigvalues)
        elif power == math.inf:
            powered_eigvalues = gs.exp(eigvalues)
        else:
            powered_eigvalues = eigvalues**power
        transp_powered_eigvalues = gs.transpose(powered_eigvalues, (0, 2, 1))
        ones = gs.ones((n_base_points, 1, self.n))
        transp_ones = gs.transpose(ones, (0, 2, 1))

        vertical_index = gs.matmul(transp_eigvalues, ones)
        horizontal_index = gs.matmul(transp_ones, eigvalues)
        one_matrix = gs.matmul(transp_ones, ones)
        vertical_index_power = gs.matmul(transp_powered_eigvalues, ones)
        horizontal_index_power = gs.matmul(transp_ones, powered_eigvalues)
        denominator = vertical_index - horizontal_index
        numerator = vertical_index_power - horizontal_index_power

        if power == 0:
            numerator = gs.where(denominator == 0, one_matrix, numerator)
            denominator = gs.where(denominator == 0, vertical_index,
                                   denominator)
        elif power == math.inf:
            numerator = gs.where(denominator == 0, vertical_index_power,
                                 numerator)
            denominator = gs.where(denominator == 0, one_matrix, denominator)
        else:
            numerator = gs.where(
                denominator == 0,
                power * vertical_index_power,
                numerator)
            denominator = gs.where(
                denominator == 0,
                vertical_index,
                denominator)

        transp_eigvectors = gs.transpose(eigvectors, (0, 2, 1))
        temp_result = gs.matmul(transp_eigvectors, tangent_vec)
        temp_result = gs.matmul(temp_result, eigvectors)
        return (eigvectors, transp_eigvectors, numerator, denominator,
                temp_result)

    def differential_power(self, power, tangent_vec, base_point):
        """Compute the differential of the matrix power function.

        Computes the differential of the power function on SPD
        matrices (A^p=exp(p log(A))) at base_point applied to
        tangent_vec.

        Parameters
        ----------
        power : int
        tangent_vec : array_like, shape=[n_samples, n, n]
            Tangent vectors.
        base_point : array_like, shape=[n_samples, n, n]
            Base points.

        Returns
        -------
        differential_power : array-like, shape=[n_samples, n, n]
        """
        eigvectors, transp_eigvectors, numerator, denominator, temp_result =\
            self.aux_differential_power(power, tangent_vec, base_point)
        power_operator = numerator / denominator
        result = power_operator * temp_result
        result = gs.matmul(result, transp_eigvectors)
        result = gs.matmul(eigvectors, result)
        return result

    def inverse_differential_power(self, power, tangent_vec, base_point):
        """Compute the inverse of the differential of the matrix power.

        Computes the inverse of the differential of the power
        function on SPD matrices (A^p=exp(p log(A))) at base_point
        applied to tangent_vec.

        Parameters
        ----------
        power : int
        tangent_vec : array_like, shape=[n_samples, n, n]
            Tangent vectors.
        base_point : array_like, shape=[n_samples, n, n]
            Base points.

        Returns
        -------
        inverse_differential_power : array-like, shape=[n_samples, n, n]
        """
        eigvectors, transp_eigvectors, numerator, denominator, temp_result =\
            self.aux_differential_power(power, tangent_vec, base_point)
        power_operator = denominator / numerator
        result = power_operator * temp_result
        result = gs.matmul(result, transp_eigvectors)
        result = gs.matmul(eigvectors, result)
        return result

    def differential_log(self, tangent_vec, base_point):
        """Compute the differential of the matrix logarithm.

        Computes the differential of the matrix logarithm on SPD
        matrices at base_point applied to tangent_vec.

        Parameters
        ----------
        tangent_vec : array_like, shape=[n_samples, n, n]
            Tangent vectors.
        base_point : array_like, shape=[n_samples, n, n]
            Base points.

        Returns
        -------
        differential_log : array-like, shape=[n_samples, n, n]
        """
        eigvectors, transp_eigvectors, numerator, denominator, temp_result =\
            self.aux_differential_power(0, tangent_vec, base_point)
        power_operator = numerator / denominator
        result = power_operator * temp_result
        result = gs.matmul(result, transp_eigvectors)
        result = gs.matmul(eigvectors, result)
        return result

    def inverse_differential_log(self, tangent_vec, base_point):
        """Compute the inverse of the differential of the matrix logarithm.

        Computes the inverse of the differential of the matrix
        logarithm on SPD matrices at base_point applied to
        tangent_vec.

        Parameters
        ----------
        tangent_vec : array_like, shape=[n_samples, n, n]
            Tangent vectors.
        base_point : array_like, shape=[n_samples, n, n]
            Base points.

        Returns
        -------
        inverse_differential_log : array-like, shape=[n_samples, n, n]
        """
        eigvectors, transp_eigvectors, numerator, denominator, temp_result =\
            self.aux_differential_power(0, tangent_vec, base_point)
        power_operator = denominator / numerator
        result = power_operator * temp_result
        result = gs.matmul(result, transp_eigvectors)
        result = gs.matmul(eigvectors, result)
        return result

    def differential_exp(self, tangent_vec, base_point):
        """Compute the differential of the matrix exponential.

        Computes the differential of the matrix exponential on SPD
        matrices at base_point applied to tangent_vec.

        Parameters
        ----------
        tangent_vec : array_like, shape=[n_samples, n, n]
            Tangent vectors.
        base_point : array_like, shape=[n_samples, n, n]
            Base points.

        Returns
        -------
        differential_exp : array-like, shape=[n_samples, n, n]
        """
        eigvectors, transp_eigvectors, numerator, denominator, temp_result = \
            self.aux_differential_power(math.inf, tangent_vec, base_point)
        power_operator = numerator / denominator
        result = power_operator * temp_result
        result = gs.matmul(result, transp_eigvectors)
        result = gs.matmul(eigvectors, result)
        return result

    def inverse_differential_exp(self, tangent_vec, base_point):
        """Compute the inverse of the differential of the matrix exponential.

        Computes the inverse of the differential of the matrix
        exponential on SPD matrices at base_point applied to
        tangent_vec.

        Parameters
        ----------
        tangent_vec : array_like, shape=[n_samples, n, n]
            Tangent vectors.
        base_point : array_like, shape=[n_samples, n, n]
            Base points.

        Returns
        -------
        inverse_differential_exp : array-like, shape=[n_samples, n, n]
        """
        eigvectors, transp_eigvectors, numerator, denominator, temp_result = \
            self.aux_differential_power(math.inf, tangent_vec, base_point)
        power_operator = denominator / numerator
        result = power_operator * temp_result
        result = gs.matmul(result, transp_eigvectors)
        result = gs.matmul(eigvectors, result)
        return result


class SPDMetricAffine(RiemannianMetric):
    """Class for the affine-invariant metric on the SPD manifold."""

    def __init__(self, n, power_affine=1):
        """Build the affine-invariant metric.

        Based on [TP2019]_.

        Parameters
        ----------
        n : int
            Matrix dimension.
        power_affine : int, optional
            Power transformation of the classical SPD metric.

        References
        ----------
        .. [TP2019] Thanwerdas, Pennec. "Is affine-invariance well defined on
          SPD matrices? A principled continuum of metrics" Proc. of GSI, 2019.
          https://arxiv.org/abs/1906.01349
        """
        dimension = int(n * (n + 1) / 2)
        super(SPDMetricAffine, self).__init__(
            dimension=dimension,
            signature=(dimension, 0, 0))
        self.n = n
        self.space = SPDMatrices(n)
        self.power_affine = power_affine

    def _aux_inner_product(self, tangent_vec_a, tangent_vec_b, inv_base_point):
        """Compute the inner product (auxiliary).

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[n_samples, n, n]
        tangent_vec_b : array-like, shape=[n_samples, n, n]
        inv_base_point : array-like, shape=[n_samples, n, n]

        Returns
        -------
        inner_product : array-like, shape=[n_samples, n, n]
        """
        aux_a = gs.matmul(inv_base_point, tangent_vec_a)
        aux_b = gs.matmul(inv_base_point, tangent_vec_b)
        inner_product = gs.trace(gs.matmul(aux_a, aux_b), axis1=1, axis2=2)
        return inner_product

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the affine-invariant inner product.

        Compute the inner product of tangent_vec_a and tangent_vec_b
        at point base_point using the affine invariant Riemannian metric.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[n_samples, n, n]
        tangent_vec_b : array-like, shape=[n_samples, n, n]
        base_point : array-like, shape=[n_samples, n, n]

        Returns
        -------
        inner_product : array-like, shape=[n_samples, n, n]
        """
        power_affine = self.power_affine
        tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=3)
        n_tangent_vecs_a, _, _ = tangent_vec_a.shape
        tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=3)
        n_tangent_vecs_b, _, _ = tangent_vec_b.shape

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, _, _ = base_point.shape

        spd_space = self.space

        assert (n_tangent_vecs_a == n_tangent_vecs_b == n_base_points
                or n_tangent_vecs_a == n_tangent_vecs_b and n_base_points == 1
                or n_base_points == n_tangent_vecs_a and n_tangent_vecs_b == 1
                or n_base_points == n_tangent_vecs_b and n_tangent_vecs_a == 1
                or n_tangent_vecs_a == 1 and n_tangent_vecs_b == 1
                or n_base_points == 1 and n_tangent_vecs_a == 1
                or n_base_points == 1 and n_tangent_vecs_b == 1)

        if n_tangent_vecs_a == 1:
            tangent_vec_a = gs.tile(
                tangent_vec_a,
                (gs.maximum(n_base_points, n_tangent_vecs_b), 1, 1))

        if n_tangent_vecs_b == 1:
            tangent_vec_b = gs.tile(
                tangent_vec_b,
                (gs.maximum(n_base_points, n_tangent_vecs_a), 1, 1))

        if n_base_points == 1:
            base_point = gs.tile(
                base_point,
                (gs.maximum(n_tangent_vecs_a, n_tangent_vecs_b), 1, 1))

        if power_affine == 1:
            inv_base_point = gs.linalg.inv(base_point)
            inner_product = self._aux_inner_product(tangent_vec_a,
                                                    tangent_vec_b,
                                                    inv_base_point)
        else:
            modified_tangent_vec_a =\
                spd_space.differential_power(power_affine, tangent_vec_a,
                                             base_point)
            modified_tangent_vec_b =\
                spd_space.differential_power(power_affine, tangent_vec_b,
                                             base_point)
            power_inv_base_point = gs.linalg.powerm(base_point, -power_affine)
            inner_product = self._aux_inner_product(modified_tangent_vec_a,
                                                    modified_tangent_vec_b,
                                                    power_inv_base_point)
            inner_product = inner_product / (power_affine**2)

        inner_product = gs.to_ndarray(inner_product, to_ndim=2, axis=1)

        return inner_product

    def _aux_exp(self, tangent_vec, sqrt_base_point, inv_sqrt_base_point):
        """Compute the exponential map (auxiliary function).

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, n, n]
        sqrt_base_point
        inv_sqrt_base_point

        Returns
        -------
        exp
        """
        tangent_vec_at_id = gs.matmul(inv_sqrt_base_point,
                                      tangent_vec)
        tangent_vec_at_id = gs.matmul(tangent_vec_at_id,
                                      inv_sqrt_base_point)
        tangent_vec_at_id = GeneralLinear.make_symmetric(tangent_vec_at_id)
        exp_from_id = gs.linalg.expm(tangent_vec_at_id)

        exp = gs.matmul(exp_from_id, sqrt_base_point)
        exp = gs.matmul(sqrt_base_point, exp)
        return exp

    def exp(self, tangent_vec, base_point):
        """Compute the affine-invariant exponential map.

        Compute the Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the metric defined in inner_product.
        This gives a symmetric positive definite matrix.

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, n, n]
        base_point : array-like, shape=[n_samples, n, n]

        Returns
        -------
        exp : array-like, shape=[n_samples, n, n]
        """
        power_affine = self.power_affine
        ndim = gs.maximum(gs.ndim(tangent_vec), gs.ndim(base_point))
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
        n_tangent_vecs, _, _ = tangent_vec.shape

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, mat_dim, _ = base_point.shape

        assert (n_tangent_vecs == n_base_points
                or n_tangent_vecs == 1
                or n_base_points == 1)

        if n_tangent_vecs == 1:
            tangent_vec = gs.tile(tangent_vec, (n_base_points, 1, 1))
        if n_base_points == 1:
            base_point = gs.tile(base_point, (n_tangent_vecs, 1, 1))

        if power_affine == 1:
            sqrt_base_point = gs.linalg.powerm(base_point, 1. / 2)
            inv_sqrt_base_point = gs.linalg.powerm(sqrt_base_point, -1)
            exp = self._aux_exp(tangent_vec, sqrt_base_point,
                                inv_sqrt_base_point)
        else:
            modified_tangent_vec = self.space.differential_power(power_affine,
                                                                 tangent_vec,
                                                                 base_point)
            power_sqrt_base_point = gs.linalg.powerm(base_point,
                                                     power_affine / 2)
            power_inv_sqrt_base_point = gs.linalg.inv(power_sqrt_base_point)
            exp = self._aux_exp(modified_tangent_vec, power_sqrt_base_point,
                                power_inv_sqrt_base_point)
            exp = gs.linalg.powerm(exp, 1 / power_affine)

        if ndim == 2:
            return exp[0]
        return exp

    def _aux_log(self, point, sqrt_base_point, inv_sqrt_base_point):
        """Compute the log (auxiliary function).

        Parameters
        ----------
        point
        sqrt_base_point
        inv_sqrt_base_point

        Returns
        -------
        log
        """
        point_near_id = gs.matmul(inv_sqrt_base_point, point)
        point_near_id = gs.matmul(point_near_id, inv_sqrt_base_point)
        point_near_id = GeneralLinear.make_symmetric(point_near_id)
        log_at_id = gs.linalg.logm(point_near_id)

        log = gs.matmul(sqrt_base_point, log_at_id)
        log = gs.matmul(log, sqrt_base_point)
        return log

    def log(self, point, base_point):
        """Compute the affine-invariant logarithm map.

        Compute the Riemannian logarithm at point base_point,
        of point wrt the metric defined in inner_product.
        This gives a tangent vector at point base_point.

        Parameters
        ----------
        point : array-like, shape=[n_samples, n, n]
        base_point : array-like, shape=[n_samples, n, n]

        Returns
        -------
        log : array-like, shape=[n_samples, n, n]
        """
        power_affine = self.power_affine
        ndim = gs.maximum(gs.ndim(point), gs.ndim(base_point))
        point = gs.to_ndarray(point, to_ndim=3)
        n_points, _, _ = point.shape

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, mat_dim, _ = base_point.shape

        assert (n_points == n_base_points
                or n_points == 1
                or n_base_points == 1)

        if n_points == 1:
            point = gs.tile(point, (n_base_points, 1, 1))
        if n_base_points == 1:
            base_point = gs.tile(base_point, (n_points, 1, 1))

        if power_affine == 1:
            sqrt_base_point = gs.linalg.powerm(base_point, 1. / 2)
            inv_sqrt_base_point = gs.linalg.powerm(sqrt_base_point, -1)
            log = self._aux_log(point, sqrt_base_point, inv_sqrt_base_point)
        else:
            power_point = gs.linalg.powerm(point, power_affine)
            power_sqrt_base_point = gs.linalg.powerm(
                base_point, power_affine / 2)
            power_inv_sqrt_base_point = gs.linalg.inv(power_sqrt_base_point)
            log = self._aux_log(
                power_point,
                power_sqrt_base_point,
                power_inv_sqrt_base_point)
            log = self.space.inverse_differential_power(power_affine, log,
                                                        base_point)

        if ndim == 2:
            return log[0]
        return log

    def geodesic(self, initial_point, initial_tangent_vec):
        """Compute the affine-invariant geodesic.

        Parameters
        ----------
        initial_point
        initial_tangent_vec

        Returns
        -------
        geodesic
        """
        return super(SPDMetricAffine, self).geodesic(
            initial_point=initial_point,
            initial_tangent_vec=initial_tangent_vec,
            point_type='matrix')

    def parallel_transport(self, tangent_vec_a, tangent_vec_b, base_point):
        r"""Parallel transport of a tangent vector.

        Closed-form solution for the parallel transport of a tangent vector a
        along the geodesic defined by exp_(base_point)(tangent_vec_b).
        Denoting `tangent_vec_a` by `S`, `base_point` by `A`, let
        `B = Exp_A(tangent_vec_b)` and :math: `E = (BA^{- 1})^({ 1 / 2})`.
        Then the
        parallel transport to `B`is:

        ..math::
                        S' = ESE^T

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[n_samples, dimension + 1]
            Tangent vector at base point to be transported.
        tangent_vec_b : array-like, shape=[n_samples, dimension + 1]
            Tangent vector at base point, initial speed of the geodesic along
            which the parallel transport is computed.
        base_point : array-like, shape=[n_samples, dimension + 1]
            point on the manifold of SPD matrices

        Returns
        -------
        transported_tangent_vec: array-like, shape=[n_samples, dimension + 1]
            Transported tangent vector at exp_(base_point)(tangent_vec_b).
        """
        end_point = self.exp(tangent_vec_b, base_point)
        inverse_base_point = GeneralLinear.inv(base_point)
        congruence_mat = GeneralLinear.mul(end_point, inverse_base_point)
        congruence_mat = gs.linalg.sqrtm(congruence_mat)
        return GeneralLinear.congruent(tangent_vec_a, congruence_mat)


class SPDMetricProcrustes(RiemannianMetric):
    """Class for the Procrustes metric on the SPD manifold.

    Based on [BJL2017].

    References
    ----------
    .. [BJL2017]_ Bhatia, Jain, Lim. "On the Bures-Wasserstein distance between
      positive definite matrices" Elsevier, Expositiones Mathematicae,
      vol. 37(2), 165-191, 2017. https://arxiv.org/pdf/1712.01504.pdf
    """

    def __init__(self, n):
        dimension = int(n * (n + 1) / 2)
        super(SPDMetricProcrustes, self).__init__(
            dimension=dimension,
            signature=(dimension, 0, 0))
        self.n = n
        self.space = SPDMatrices(n)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the Procrustes inner product.

        Compute the inner product of tangent_vec_a and tangent_vec_b
        at point base_point using the Procrustes Riemannian metric.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[n_samples, n, n]
        tangent_vec_b : array-like, shape=[n_samples, n, n]
        base_point : array-like, shape=[n_samples, n, n]

        Returns
        -------
        inner_product : float
        """
        tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=3)
        n_tangent_vecs_a, _, _ = tangent_vec_a.shape
        tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=3)
        n_tangent_vecs_b, _, _ = tangent_vec_b.shape

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, _, _ = base_point.shape

        assert (n_tangent_vecs_a == n_tangent_vecs_b == n_base_points
                or n_tangent_vecs_a == n_tangent_vecs_b and n_base_points == 1
                or n_base_points == n_tangent_vecs_a and n_tangent_vecs_b == 1
                or n_base_points == n_tangent_vecs_b and n_tangent_vecs_a == 1
                or n_tangent_vecs_a == 1 and n_tangent_vecs_b == 1
                or n_base_points == 1 and n_tangent_vecs_a == 1
                or n_base_points == 1 and n_tangent_vecs_b == 1)

        if n_tangent_vecs_a == 1:
            tangent_vec_a = gs.tile(
                tangent_vec_a,
                (gs.maximum(n_base_points, n_tangent_vecs_b), 1, 1))

        if n_tangent_vecs_b == 1:
            tangent_vec_b = gs.tile(
                tangent_vec_b,
                (gs.maximum(n_base_points, n_tangent_vecs_a), 1, 1))

        if n_base_points == 1:
            base_point = gs.tile(
                base_point,
                (gs.maximum(n_tangent_vecs_a, n_tangent_vecs_b), 1, 1))

        spd_space = self.space
        modified_tangent_vec_a =\
            spd_space.inverse_differential_power(2, tangent_vec_a, base_point)
        product = gs.matmul(modified_tangent_vec_a, tangent_vec_b)
        result = gs.trace(product, axis1=1, axis2=2) / 2
        return result


class SPDMetricEuclidean(RiemannianMetric):
    """Class for the Euclidean metric on the SPD manifold."""

    def __init__(self, n, power_euclidean=1):
        dimension = int(n * (n + 1) / 2)
        super(SPDMetricEuclidean, self).__init__(
            dimension=dimension,
            signature=(dimension, 0, 0))
        self.n = n
        self.space = SPDMatrices(n)
        self.power_euclidean = power_euclidean

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the Euclidean inner product.

        Compute the inner product of tangent_vec_a and tangent_vec_b
        at point base_point using the power-Euclidean metric.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[n_samples, n, n]
        tangent_vec_b : array-like, shape=[n_samples, n, n]
        base_point : array-like, shape=[n_samples, n, n]

        Returns
        -------
        inner_product : float
        """
        power_euclidean = self.power_euclidean
        tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=3)
        n_tangent_vecs_a, _, _ = tangent_vec_a.shape
        tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=3)
        n_tangent_vecs_b, _, _ = tangent_vec_b.shape

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, _, _ = base_point.shape

        spd_space = self.space

        assert (n_tangent_vecs_a == n_tangent_vecs_b == n_base_points
                or n_tangent_vecs_a == n_tangent_vecs_b and n_base_points == 1
                or n_base_points == n_tangent_vecs_a and n_tangent_vecs_b == 1
                or n_base_points == n_tangent_vecs_b and n_tangent_vecs_a == 1
                or n_tangent_vecs_a == 1 and n_tangent_vecs_b == 1
                or n_base_points == 1 and n_tangent_vecs_a == 1
                or n_base_points == 1 and n_tangent_vecs_b == 1)

        if n_tangent_vecs_a == 1:
            tangent_vec_a = gs.tile(
                tangent_vec_a,
                (gs.maximum(n_base_points, n_tangent_vecs_b), 1, 1))

        if n_tangent_vecs_b == 1:
            tangent_vec_b = gs.tile(
                tangent_vec_b,
                (gs.maximum(n_base_points, n_tangent_vecs_a), 1, 1))

        if n_base_points == 1:
            base_point = gs.tile(
                base_point,
                (gs.maximum(n_tangent_vecs_a, n_tangent_vecs_b), 1, 1))

        if power_euclidean == 1:
            product = gs.matmul(tangent_vec_a, tangent_vec_b)
            inner_product = gs.trace(product, axis1=1, axis2=2)
        else:
            modified_tangent_vec_a = \
                spd_space.differential_power(power_euclidean, tangent_vec_a,
                                             base_point)
            modified_tangent_vec_b = \
                spd_space.differential_power(power_euclidean, tangent_vec_b,
                                             base_point)
            product = gs.matmul(modified_tangent_vec_a, modified_tangent_vec_b)
            inner_product = gs.trace(product, axis1=1, axis2=2) \
                / (power_euclidean ** 2)

        inner_product = gs.to_ndarray(inner_product, to_ndim=2, axis=1)

        return inner_product

    def exp_domain(self, tangent_vec, base_point):
        """Compute the domain of the Euclidean exponential map.

        Compute the real interval of time where the Euclidean geodesic starting
        at point `base_point` in direction `tangent_vec` is defined.

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, n, n]
        base_point : array-like, shape=[n_samples, n, n]

        Returns
        -------
        exp_domain : array-like, shape=[n_samples, 2]
        """
        base_point = gs.to_ndarray(base_point, to_ndim=3)
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
        invsqrt_base_point = gs.linalg.powerm(base_point, -.5)
        reduced_vec = gs.matmul(invsqrt_base_point, tangent_vec)
        reduced_vec = gs.matmul(reduced_vec, invsqrt_base_point)
        eigvals = gs.linalg.eigvalsh(reduced_vec)
        min_eig = gs.amin(eigvals, axis=1)
        max_eig = gs.amax(eigvals, axis=1)
        inf_value = gs.where(max_eig <= 0, -math.inf, - 1 / max_eig)
        inf_value = gs.to_ndarray(inf_value, to_ndim=2)
        sup_value = gs.where(min_eig >= 0, math.inf, - 1 / min_eig)
        sup_value = gs.to_ndarray(sup_value, to_ndim=2)
        domain = gs.concatenate((inf_value, sup_value), axis=1)

        return domain


class SPDMetricLogEuclidean(RiemannianMetric):
    """Class for the Log-Euclidean metric on the SPD manifold."""

    def __init__(self, n):
        dimension = int(n * (n + 1) / 2)
        super(SPDMetricLogEuclidean, self).__init__(
            dimension=dimension,
            signature=(dimension, 0, 0))
        self.n = n
        self.space = SPDMatrices(n)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the Log-Euclidean inner product.

        Compute the inner product of tangent_vec_a and tangent_vec_b
        at point base_point using the log-Euclidean metric.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[n_samples, n, n]
        tangent_vec_b : array-like, shape=[n_samples, n, n]
        base_point : array-like, shape=[n_samples, n, n]

        Returns
        -------
        inner_product : float
        """
        tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=3)
        n_tangent_vecs_a, _, _ = tangent_vec_a.shape
        tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=3)
        n_tangent_vecs_b, _, _ = tangent_vec_b.shape

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_base_points, _, _ = base_point.shape

        spd_space = self.space

        assert (n_tangent_vecs_a == n_tangent_vecs_b == n_base_points
                or n_tangent_vecs_a == n_tangent_vecs_b and n_base_points == 1
                or n_base_points == n_tangent_vecs_a and n_tangent_vecs_b == 1
                or n_base_points == n_tangent_vecs_b and n_tangent_vecs_a == 1
                or n_tangent_vecs_a == 1 and n_tangent_vecs_b == 1
                or n_base_points == 1 and n_tangent_vecs_a == 1
                or n_base_points == 1 and n_tangent_vecs_b == 1)

        if n_tangent_vecs_a == 1:
            tangent_vec_a = gs.tile(
                tangent_vec_a,
                (gs.maximum(n_base_points, n_tangent_vecs_b), 1, 1))

        if n_tangent_vecs_b == 1:
            tangent_vec_b = gs.tile(
                tangent_vec_b,
                (gs.maximum(n_base_points, n_tangent_vecs_a), 1, 1))

        if n_base_points == 1:
            base_point = gs.tile(
                base_point,
                (gs.maximum(n_tangent_vecs_a, n_tangent_vecs_b), 1, 1))

        modified_tangent_vec_a = spd_space.differential_log(tangent_vec_a,
                                                            base_point)
        modified_tangent_vec_b = spd_space.differential_log(tangent_vec_b,
                                                            base_point)
        product = gs.matmul(modified_tangent_vec_a, modified_tangent_vec_b)
        inner_product = gs.trace(product, axis1=1, axis2=2)

        inner_product = gs.to_ndarray(inner_product, to_ndim=2, axis=1)

        return inner_product

    def exp(self, tangent_vec, base_point):
        """Compute the Log-Euclidean exponential map.

        Compute the Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the Log-Euclidean metric.
        This gives a symmetric positive definite matrix.

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, n, n]
        base_point : array-like, shape=[n_samples, n, n]

        Returns
        -------
        exp : array-like, shape=[n_samples, n, n]
        """
        ndim = gs.maximum(gs.ndim(tangent_vec), gs.ndim(base_point))
        log_base_point = gs.linalg.logm(base_point)
        dlog_tangent_vec = self.space.differential_log(tangent_vec, base_point)
        exp = gs.linalg.expm(log_base_point + dlog_tangent_vec)

        if ndim == 2:
            return exp[0]
        return exp

    def log(self, point, base_point):
        """Compute the Log-Euclidean logarithm map.

        Compute the Riemannian logarithm at point base_point,
        of point wrt the Log-Euclidean metric.
        This gives a tangent vector at point base_point.

        Parameters
        ----------
        point : array-like, shape=[n_samples, n, n]
        base_point : array-like, shape=[n_samples, n, n]

        Returns
        -------
        log : array-like, shape=[n_samples, n, n]
        """
        ndim = gs.maximum(gs.ndim(point), gs.ndim(base_point))
        log_base_point = gs.linalg.logm(base_point)
        log_point = gs.linalg.logm(point)
        log = self.space.differential_exp(log_point - log_base_point,
                                          log_base_point)

        if ndim == 2:
            return log[0]
        return log

    def geodesic(self, initial_point, initial_tangent_vec):
        """Compute the Log-Euclidean geodesic.

        Parameters
        ----------
        initial_point : array-like, shape=[n_samples, n, n]
        initial_tangent_vec : array-like, shape=[n_samples, n, n]

        Returns
        -------
        path : callable
            The time parameterized geodesic.
        """
        def path(t):
            return self.exp(t * initial_tangent_vec, initial_point)

        return path
