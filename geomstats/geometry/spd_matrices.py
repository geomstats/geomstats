"""The manifold of symmetric positive definite (SPD) matrices."""

import math

import geomstats.backend as gs
import geomstats.vectorization
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.symmetric_matrices import SymmetricMatrices

EPSILON = 1e-6
TOLERANCE = 1e-12


class SPDMatrices(SymmetricMatrices, EmbeddedManifold):
    """Class for the manifold of symmetric positive definite (SPD) matrices.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n):
        super(SPDMatrices, self).__init__(
            n=n,
            dim=int(n * (n + 1) / 2),
            embedding_manifold=GeneralLinear(n=n))

    def belongs(self, mat, atol=TOLERANCE):
        """Check if a matrix is symmetric and invertible.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix to be checked.
        atol : float
            Tolerance.
            Optional, default: TOLERANCE.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if mat is an SPD matrix.
        """
        is_symmetric = super(SPDMatrices, self).belongs(mat, atol)
        eigvalues, _ = gs.linalg.eigh(mat)
        is_positive = gs.all(eigvalues > 0, axis=-1)
        belongs = gs.logical_and(is_symmetric, is_positive)
        return belongs

    def random_uniform(self, n_samples=1):
        """Sample in SPD(n) from the log-uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Points sampled in SPD(n).
        """
        n = self.n
        size = (n_samples, n, n) if n_samples != 1 else (n, n)

        mat = 2 * gs.random.rand(*size) - 1
        spd_mat = GeneralLinear.exp(mat + Matrices.transpose(mat))

        return spd_mat

    def random_tangent_vec_uniform(self, n_samples=1, base_point=None):
        """Sample on the tangent space of SPD(n) from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        base_point : array-like, shape=[..., n, n]
            Base point of the tangent space.
            Optional, default: None.

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Points sampled in the tangent space at base_point.
        """
        n = self.n
        size = (n_samples, n, n) if n_samples != 1 else (n, n)

        if base_point is None:
            base_point = gs.eye(n)

        sqrt_base_point = gs.linalg.sqrtm(base_point)

        tangent_vec_at_id = 2 * gs.random.rand(*size) - 1
        tangent_vec_at_id += Matrices.transpose(tangent_vec_at_id)

        tangent_vec = gs.einsum(
            '...ij,...jk->...ik', sqrt_base_point, tangent_vec_at_id)
        tangent_vec = gs.einsum(
            '...ij,...jk->...ik', tangent_vec, sqrt_base_point)

        return tangent_vec

    @staticmethod
    @geomstats.vectorization.decorator(['else', 'matrix', 'matrix'])
    def aux_differential_power(power, tangent_vec, base_point):
        """Compute the differential of the matrix power.

        Auxiliary function to the functions differential_power and
        inverse_differential_power.

        Parameters
        ----------
        power : float
            Power function to differentiate.
        tangent_vec : array_like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array_like, shape=[..., n, n]
            Base point.

        Returns
        -------
        eigvectors : array-like, shape=[..., n, n]
        transp_eigvectors : array-like, shape=[..., n, n]
        numerator : array-like, shape=[..., n, n]
        denominator : array-like, shape=[..., n, n]
        temp_result : array-like, shape=[..., n, n]
        """
        n_tangent_vecs, _, _ = tangent_vec.shape
        n_base_points, _, n = base_point.shape

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
        ones = gs.ones((n_base_points, 1, n))
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

        if n_base_points == n_tangent_vecs == 1:
            transp_eigvectors = gs.squeeze(transp_eigvectors, axis=0)
            eigvectors = gs.squeeze(eigvectors, axis=0)
            temp_result = gs.squeeze(temp_result, axis=0)
            numerator = gs.squeeze(numerator, axis=0)
            denominator = gs.squeeze(denominator, axis=0)

        return (eigvectors, transp_eigvectors, numerator, denominator,
                temp_result)

    @classmethod
    @geomstats.vectorization.decorator(['else', 'else', 'matrix', 'matrix'])
    def differential_power(cls, power, tangent_vec, base_point):
        r"""Compute the differential of the matrix power function.

        Compute the differential of the power function on SPD(n)
        (:math: `A^p=\exp(p \log(A))`) at base_point applied to tangent_vec.

        Parameters
        ----------
        power : int
            Power.
        tangent_vec : array_like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array_like, shape=[..., n, n]
            Base point.

        Returns
        -------
        differential_power : array-like, shape=[..., n, n]
            Differential of the power function.
        """
        eigvectors, transp_eigvectors, numerator, denominator, temp_result =\
            cls.aux_differential_power(power, tangent_vec, base_point)
        power_operator = numerator / denominator
        result = power_operator * temp_result
        result = gs.matmul(result, transp_eigvectors)
        result = gs.matmul(eigvectors, result)
        return result

    @classmethod
    @geomstats.vectorization.decorator(['else', 'else', 'matrix', 'matrix'])
    def inverse_differential_power(cls, power, tangent_vec, base_point):
        r"""Compute the inverse of the differential of the matrix power.

        Compute the inverse of the differential of the power
        function on SPD matrices (:math: `A^p=exp(p log(A))`) at base_point
        applied to tangent_vec.

        Parameters
        ----------
        power : int
            Power.
        tangent_vec : array_like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array_like, shape=[..., n, n]
            Base point.

        Returns
        -------
        inverse_differential_power : array-like, shape=[..., n, n]
            Inverse of the differential of the power function.
        """
        eigvectors, transp_eigvectors, numerator, denominator, temp_result =\
            cls.aux_differential_power(power, tangent_vec, base_point)
        power_operator = denominator / numerator
        result = power_operator * temp_result
        result = gs.matmul(result, transp_eigvectors)
        result = gs.matmul(eigvectors, result)
        return result

    @classmethod
    @geomstats.vectorization.decorator(['else', 'matrix', 'matrix'])
    def differential_log(cls, tangent_vec, base_point):
        """Compute the differential of the matrix logarithm.

        Compute the differential of the matrix logarithm on SPD
        matrices at base_point applied to tangent_vec.

        Parameters
        ----------
        tangent_vec : array_like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array_like, shape=[..., n, n]
            Base point.

        Returns
        -------
        differential_log : array-like, shape=[..., n, n]
            Differential of the matrix logarithm.
        """
        eigvectors, transp_eigvectors, numerator, denominator, temp_result =\
            cls.aux_differential_power(0, tangent_vec, base_point)
        power_operator = numerator / denominator
        result = power_operator * temp_result
        result = gs.matmul(result, transp_eigvectors)
        result = gs.matmul(eigvectors, result)
        return result

    @classmethod
    @geomstats.vectorization.decorator(['else', 'matrix', 'matrix'])
    def inverse_differential_log(cls, tangent_vec, base_point):
        """Compute the inverse of the differential of the matrix logarithm.

        Compute the inverse of the differential of the matrix
        logarithm on SPD matrices at base_point applied to
        tangent_vec.

        Parameters
        ----------
        tangent_vec : array_like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array_like, shape=[..., n, n]
            Base point.

        Returns
        -------
        inverse_differential_log : array-like, shape=[..., n, n]
            Inverse of the differential of the matrix logarithm.
        """
        eigvectors, transp_eigvectors, numerator, denominator, temp_result =\
            cls.aux_differential_power(0, tangent_vec, base_point)
        power_operator = denominator / numerator
        result = power_operator * temp_result
        result = gs.matmul(result, transp_eigvectors)
        result = gs.matmul(eigvectors, result)
        return result

    @classmethod
    @geomstats.vectorization.decorator(['else', 'matrix', 'matrix'])
    def differential_exp(cls, tangent_vec, base_point):
        """Compute the differential of the matrix exponential.

        Computes the differential of the matrix exponential on SPD
        matrices at base_point applied to tangent_vec.

        Parameters
        ----------
        tangent_vec : array_like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array_like, shape=[..., n, n]
            Base point.

        Returns
        -------
        differential_exp : array-like, shape=[..., n, n]
            Differential of the matrix exponential.
        """
        eigvectors, transp_eigvectors, numerator, denominator, temp_result = \
            cls.aux_differential_power(math.inf, tangent_vec, base_point)
        power_operator = numerator / denominator
        result = power_operator * temp_result
        result = gs.matmul(result, transp_eigvectors)
        result = gs.matmul(eigvectors, result)
        return result

    @classmethod
    @geomstats.vectorization.decorator(['else', 'matrix', 'matrix'])
    def inverse_differential_exp(cls, tangent_vec, base_point):
        """Compute the inverse of the differential of the matrix exponential.

        Computes the inverse of the differential of the matrix
        exponential on SPD matrices at base_point applied to
        tangent_vec.

        Parameters
        ----------
        tangent_vec : array_like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array_like, shape=[..., n, n]
            Base point.

        Returns
        -------
        inverse_differential_exp : array-like, shape=[..., n, n]
            Inverse of the differential of the matrix exponential.
        """
        eigvectors, transp_eigvectors, numerator, denominator, temp_result = \
            cls.aux_differential_power(math.inf, tangent_vec, base_point)
        power_operator = denominator / numerator
        result = power_operator * temp_result
        result = gs.matmul(result, transp_eigvectors)
        result = gs.matmul(eigvectors, result)
        return result

    @classmethod
    def logm(cls, mat):
        """
        Compute the matrix log for a symmetric matrix.

        Parameters
        ----------
        mat : array_like, shape=[..., n, n]
            Symmetric matrix.

        Returns
        -------
        log : array_like, shape=[..., n, n]
            Matrix logarithm of mat.
        """
        return cls.apply_func_to_eigvals(mat, gs.log, check_positive=True)


class SPDMetricAffine(RiemannianMetric):
    """Class for the affine-invariant metric on the SPD manifold."""

    def __init__(self, n, power_affine=1):
        """Build the affine-invariant metric.

        Parameters
        ----------
        n : int
            Integer representing the shape of the matrices: n x n.
        power_affine : int
            Power transformation of the classical SPD metric.
            Optional, default: 1.

        References
        ----------
        .. [TP2019] Thanwerdas, Pennec. "Is affine-invariance well defined on
          SPD matrices? A principled continuum of metrics" Proc. of GSI, 2019.
          https://arxiv.org/abs/1906.01349
        """
        dim = int(n * (n + 1) / 2)
        super(SPDMetricAffine, self).__init__(
            dim=dim,
            signature=(dim, 0, 0),
            default_point_type='matrix')
        self.n = n
        self.space = SPDMatrices(n)
        self.power_affine = power_affine

    @staticmethod
    def _aux_inner_product(tangent_vec_a, tangent_vec_b, inv_base_point):
        """Compute the inner-product (auxiliary).

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
        tangent_vec_b : array-like, shape=[..., n, n]
        inv_base_point : array-like, shape=[..., n, n]

        Returns
        -------
        inner_product : array-like, shape=[..., n, n]
        """
        aux_a = gs.einsum(
            '...ij,...jk->...ik', inv_base_point, tangent_vec_a)
        aux_b = gs.einsum(
            '...ij,...jk->...ik', inv_base_point, tangent_vec_b)
        prod = gs.einsum(
            '...ij,...jk->...ik', aux_a, aux_b)
        inner_product = gs.trace(prod, axis1=-2, axis2=-1)
        return inner_product

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the affine-invariant inner-product.

        Compute the inner-product of tangent_vec_a and tangent_vec_b
        at point base_point using the affine invariant Riemannian metric.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        inner_product : array-like, shape=[..., n, n]
            Inner-product.
        """
        power_affine = self.power_affine
        spd_space = self.space

        if power_affine == 1:
            inv_base_point = GeneralLinear.inverse(base_point)
            inner_product = self._aux_inner_product(
                tangent_vec_a, tangent_vec_b, inv_base_point)
        else:
            modified_tangent_vec_a = spd_space.differential_power(
                power_affine, tangent_vec_a, base_point)
            modified_tangent_vec_b = spd_space.differential_power(
                power_affine, tangent_vec_b, base_point)
            power_inv_base_point = SymmetricMatrices.powerm(
                base_point, -power_affine)
            inner_product = self._aux_inner_product(
                modified_tangent_vec_a,
                modified_tangent_vec_b,
                power_inv_base_point)

            inner_product = inner_product / (power_affine**2)

        return inner_product

    @staticmethod
    def _aux_exp(tangent_vec, sqrt_base_point, inv_sqrt_base_point):
        """Compute the exponential map (auxiliary function).

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
        sqrt_base_point : array-like, shape=[..., n, n]
        inv_sqrt_base_point : array-like, shape=[..., n, n]

        Returns
        -------
        exp : array-like, shape=[..., n, n]
        """
        tangent_vec_at_id = gs.einsum(
            '...ij,...jk->...ik', inv_sqrt_base_point, tangent_vec)
        tangent_vec_at_id = gs.einsum(
            '...ij,...jk->...ik', tangent_vec_at_id, inv_sqrt_base_point)
        tangent_vec_at_id = GeneralLinear.to_symmetric(tangent_vec_at_id)
        exp_from_id = SymmetricMatrices.expm(tangent_vec_at_id)

        exp = gs.einsum(
            '...ij,...jk->...ik', exp_from_id, sqrt_base_point)
        exp = gs.einsum(
            '...ij,...jk->...ik', sqrt_base_point, exp)
        return exp

    def exp(self, tangent_vec, base_point):
        """Compute the affine-invariant exponential map.

        Compute the Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the metric defined in inner_product.
        This gives a symmetric positive definite matrix.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        exp : array-like, shape=[..., n, n]
            Riemannian exponential.
        """
        power_affine = self.power_affine

        if power_affine == 1:
            sqrt_base_point = SymmetricMatrices.powerm(base_point, 1. / 2)
            inv_sqrt_base_point = SymmetricMatrices.powerm(sqrt_base_point, -1)
            exp = self._aux_exp(
                tangent_vec, sqrt_base_point, inv_sqrt_base_point)
        else:
            modified_tangent_vec = self.space.differential_power(
                power_affine, tangent_vec, base_point)
            power_sqrt_base_point = SymmetricMatrices.powerm(
                base_point, power_affine / 2)
            power_inv_sqrt_base_point = GeneralLinear.inverse(
                power_sqrt_base_point)
            exp = self._aux_exp(
                modified_tangent_vec,
                power_sqrt_base_point,
                power_inv_sqrt_base_point)
            exp = SymmetricMatrices.powerm(exp, 1 / power_affine)

        return exp

    @staticmethod
    def _aux_log(point, sqrt_base_point, inv_sqrt_base_point):
        """Compute the log (auxiliary function).

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
        sqrt_base_point : array-like, shape=[..., n, n]
        inv_sqrt_base_point : array-like, shape=[.., n, n]

        Returns
        -------
        log : array-like, shape=[..., n, n]
        """
        point_near_id = gs.einsum(
            '...ij,...jk->...ik', inv_sqrt_base_point, point)
        point_near_id = gs.einsum(
            '...ij,...jk->...ik', point_near_id, inv_sqrt_base_point)
        point_near_id = GeneralLinear.to_symmetric(point_near_id)
        log_at_id = SPDMatrices.logm(point_near_id)

        log = gs.einsum(
            '...ij,...jk->...ik', sqrt_base_point, log_at_id)
        log = gs.einsum(
            '...ij,...jk->...ik', log, sqrt_base_point)
        return log

    def log(self, point, base_point):
        """Compute the affine-invariant logarithm map.

        Compute the Riemannian logarithm at point base_point,
        of point wrt the metric defined in inner_product.
        This gives a tangent vector at point base_point.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        log : array-like, shape=[..., n, n]
            Riemannian logarithm of point at base_point.
        """
        power_affine = self.power_affine

        if power_affine == 1:
            sqrt_base_point = SymmetricMatrices.powerm(base_point, 1. / 2)
            inv_sqrt_base_point = SymmetricMatrices.powerm(sqrt_base_point, -1)
            log = self._aux_log(point, sqrt_base_point, inv_sqrt_base_point)
        else:
            power_point = SymmetricMatrices.powerm(point, power_affine)
            power_sqrt_base_point = SymmetricMatrices.powerm(
                base_point, power_affine / 2)
            power_inv_sqrt_base_point = gs.linalg.inv(power_sqrt_base_point)
            log = self._aux_log(
                power_point,
                power_sqrt_base_point,
                power_inv_sqrt_base_point)
            log = self.space.inverse_differential_power(power_affine, log,
                                                        base_point)
        return log

    def geodesic(self, initial_point, initial_tangent_vec):
        """Compute the affine-invariant geodesic.

        Parameters
        ----------
        initial_point : array-like, shape=[..., n, n]
            Initial point of the geodesic.
        initial_tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at the initial point, the initial speed
            of the geodesic.

        Returns
        -------
        geodesic : callable
            Time-parameterized geodesic curve.
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
        tangent_vec_a : array-like, shape=[..., dim + 1]
            Tangent vector at base point to be transported.
        tangent_vec_b : array-like, shape=[..., dim + 1]
            Tangent vector at base point, initial speed of the geodesic along
            which the parallel transport is computed.
        base_point : array-like, shape=[..., dim + 1]
            Point on the manifold of SPD matrices.

        Returns
        -------
        transported_tangent_vec: array-like, shape=[..., dim + 1]
            Transported tangent vector at exp_(base_point)(tangent_vec_b).
        """
        end_point = self.exp(tangent_vec_b, base_point)
        inverse_base_point = GeneralLinear.inverse(base_point)
        congruence_mat = GeneralLinear.mul(end_point, inverse_base_point)
        congruence_mat = gs.linalg.sqrtm(congruence_mat)
        return GeneralLinear.congruent(tangent_vec_a, congruence_mat)


class SPDMetricProcrustes(RiemannianMetric):
    """Class for the Procrustes metric on the SPD manifold.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.

    References
    ----------
    .. [BJL2017]_ Bhatia, Jain, Lim. "On the Bures-Wasserstein distance between
      positive definite matrices" Elsevier, Expositiones Mathematicae,
      vol. 37(2), 165-191, 2017. https://arxiv.org/pdf/1712.01504.pdf
    """

    def __init__(self, n):
        dim = int(n * (n + 1) / 2)
        super(SPDMetricProcrustes, self).__init__(
            dim=dim,
            signature=(dim, 0, 0),
            default_point_type='matrix')
        self.n = n
        self.space = SPDMatrices(n)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the Procrustes inner-product.

        Compute the inner-product of tangent_vec_a and tangent_vec_b
        at point base_point using the Procrustes Riemannian metric.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """
        spd_space = self.space
        modified_tangent_vec_a =\
            spd_space.inverse_differential_power(2, tangent_vec_a, base_point)
        product = gs.einsum(
            '...ij,...jk->...ik', modified_tangent_vec_a, tangent_vec_b)
        result = gs.trace(product, axis1=-2, axis2=-1) / 2
        return result


class SPDMetricEuclidean(RiemannianMetric):
    """Class for the Euclidean metric on the SPD manifold."""

    def __init__(self, n, power_euclidean=1):
        dim = int(n * (n + 1) / 2)
        super(SPDMetricEuclidean, self).__init__(
            dim=dim,
            signature=(dim, 0, 0),
            default_point_type='matrix')
        self.n = n
        self.space = SPDMatrices(n)
        self.power_euclidean = power_euclidean

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the Euclidean inner-product.

        Compute the inner-product of tangent_vec_a and tangent_vec_b
        at point base_point using the power-Euclidean metric.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """
        power_euclidean = self.power_euclidean

        spd_space = self.space

        if power_euclidean == 1:
            product = gs.einsum(
                '...ij,...jk->...ik', tangent_vec_a, tangent_vec_b)
            inner_product = gs.trace(product, axis1=-2, axis2=-1)
        else:
            modified_tangent_vec_a = spd_space.differential_power(
                power_euclidean, tangent_vec_a, base_point)
            modified_tangent_vec_b = spd_space.differential_power(
                power_euclidean, tangent_vec_b, base_point)
            product = gs.einsum(
                '...ij,...jk->...ik',
                modified_tangent_vec_a, modified_tangent_vec_b)
            inner_product = gs.trace(product, axis1=-2, axis2=-1) \
                / (power_euclidean ** 2)

        return inner_product

    @staticmethod
    @geomstats.vectorization.decorator(['matrix', 'matrix'])
    def exp_domain(tangent_vec, base_point):
        """Compute the domain of the Euclidean exponential map.

        Compute the real interval of time where the Euclidean geodesic starting
        at point `base_point` in direction `tangent_vec` is defined.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        exp_domain : array-like, shape=[..., 2]
            Interval of time where the geodesic is defined.
        """
        invsqrt_base_point = gs.linalg.powerm(base_point, -.5)

        reduced_vec = gs.matmul(invsqrt_base_point, tangent_vec)
        reduced_vec = gs.matmul(reduced_vec, invsqrt_base_point)
        eigvals = gs.linalg.eigvalsh(reduced_vec)
        min_eig = gs.amin(eigvals, axis=1)
        max_eig = gs.amax(eigvals, axis=1)
        inf_value = gs.where(
            max_eig <= 0., gs.array(-math.inf), - 1. / max_eig)
        inf_value = gs.to_ndarray(inf_value, to_ndim=2)
        sup_value = gs.where(
            min_eig >= 0., gs.array(-math.inf), - 1. / min_eig)
        sup_value = gs.to_ndarray(sup_value, to_ndim=2)
        domain = gs.concatenate((inf_value, sup_value), axis=1)

        return domain


class SPDMetricLogEuclidean(RiemannianMetric):
    """Class for the Log-Euclidean metric on the SPD manifold.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n):
        dim = int(n * (n + 1) / 2)
        super(SPDMetricLogEuclidean, self).__init__(
            dim=dim,
            signature=(dim, 0, 0),
            default_point_type='matrix')
        self.n = n
        self.space = SPDMatrices(n)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the Log-Euclidean inner-product.

        Compute the inner-product of tangent_vec_a and tangent_vec_b
        at point base_point using the log-Euclidean metric.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """
        spd_space = self.space

        modified_tangent_vec_a = spd_space.differential_log(
            tangent_vec_a, base_point)
        modified_tangent_vec_b = spd_space.differential_log(
            tangent_vec_b, base_point)
        product = gs.einsum(
            '...ij,...jk->...ik',
            modified_tangent_vec_a, modified_tangent_vec_b)
        inner_product = gs.trace(product, axis1=-2, axis2=-1)

        return inner_product

    def exp(self, tangent_vec, base_point):
        """Compute the Log-Euclidean exponential map.

        Compute the Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the Log-Euclidean metric.
        This gives a symmetric positive definite matrix.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        exp : array-like, shape=[..., n, n]
            Riemannian exponential.
        """
        log_base_point = self.space.logm(base_point)
        dlog_tangent_vec = self.space.differential_log(tangent_vec, base_point)
        exp = SymmetricMatrices.expm(log_base_point + dlog_tangent_vec)

        return exp

    def log(self, point, base_point):
        """Compute the Log-Euclidean logarithm map.

        Compute the Riemannian logarithm at point base_point,
        of point wrt the Log-Euclidean metric.
        This gives a tangent vector at point base_point.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        log : array-like, shape=[..., n, n]
            Riemannian logarithm.
        """
        log_base_point = SPDMatrices.logm(base_point)
        log_point = SPDMatrices.logm(point)
        log = self.space.differential_exp(
            log_point - log_base_point, log_base_point)

        return log

    def geodesic(self, initial_point, initial_tangent_vec):
        """Compute the Log-Euclidean geodesic.

        Parameters
        ----------
        initial_point : array-like, shape=[..., n, n]
            Initial point of the geodesic.
        initial_tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at the initial point, the initial speed
            of the geodesic.

        Returns
        -------
        geodesic : callable
            Time-parameterized geodesic curve.
        """
        def path(t):
            return self.exp(t * initial_tangent_vec, initial_point)

        return path
