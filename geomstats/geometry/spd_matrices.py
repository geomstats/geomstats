"""The manifold of symmetric positive definite (SPD) matrices.

Lead author: Yann Thanwerdas.
"""

import math

import geomstats.backend as gs
import geomstats.vectorization
from geomstats.geometry.base import OpenSet
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.geometry.positive_lower_triangular_matrices import (
    PositiveLowerTriangularMatrices,
)
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.integrator import integrate


class SPDMatrices(OpenSet):
    """Class for the manifold of symmetric positive definite (SPD) matrices.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n, **kwargs):
        kwargs.setdefault("metric", SPDAffineMetric(n))
        super().__init__(
            dim=int(n * (n + 1) / 2), embedding_space=SymmetricMatrices(n), **kwargs
        )
        self.n = n

    def belongs(self, point, atol=gs.atol):
        """Check if a matrix is symmetric with positive eigenvalues.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix to be checked.
        atol : float
            Tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if mat is an SPD matrix.
        """
        is_sym = self.embedding_space.belongs(point, atol)
        is_pd = Matrices.is_pd(point)
        belongs = gs.logical_and(is_sym, is_pd)
        return belongs

    def projection(self, point):
        """Project a matrix to the space of SPD matrices.

        First the symmetric part of point is computed, then the eigenvalues
        are floored to gs.atol.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix to project.

        Returns
        -------
        projected: array-like, shape=[..., n, n]
            SPD matrix.
        """
        sym = Matrices.to_symmetric(point)
        eigvals, eigvecs = gs.linalg.eigh(sym)
        regularized = gs.where(eigvals < gs.atol, gs.atol, eigvals)
        reconstruction = gs.einsum("...ij,...j->...ij", eigvecs, regularized)
        return Matrices.mul(reconstruction, Matrices.transpose(eigvecs))

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in SPD(n) from the log-uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample in the tangent space.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Points sampled in SPD(n).
        """
        n = self.n
        size = (n_samples, n, n) if n_samples != 1 else (n, n)

        mat = bound * (2 * gs.random.rand(*size) - 1)
        spd_mat = GeneralLinear.exp(Matrices.to_symmetric(mat))

        return spd_mat

    def random_tangent_vec(self, base_point, n_samples=1):
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

        tangent_vec_at_id_aux = 2 * gs.random.rand(*size) - 1
        tangent_vec_at_id = tangent_vec_at_id_aux + Matrices.transpose(
            tangent_vec_at_id_aux
        )

        tangent_vec = Matrices.mul(sqrt_base_point, tangent_vec_at_id, sqrt_base_point)

        return tangent_vec

    @staticmethod
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
        eigvalues, eigvectors = gs.linalg.eigh(base_point)

        if power == 0:
            powered_eigvalues = gs.log(eigvalues)
        elif power == math.inf:
            powered_eigvalues = gs.exp(eigvalues)
        else:
            powered_eigvalues = eigvalues**power

        denominator = eigvalues[..., :, None] - eigvalues[..., None, :]
        numerator = powered_eigvalues[..., :, None] - powered_eigvalues[..., None, :]

        null_denominator = gs.abs(denominator) < gs.atol
        if power == 0:
            numerator = gs.where(null_denominator, gs.ones_like(numerator), numerator)
            denominator = gs.where(
                null_denominator, eigvalues[..., :, None], denominator
            )
        elif power == math.inf:
            numerator = gs.where(
                null_denominator, powered_eigvalues[..., :, None], numerator
            )
            denominator = gs.where(
                null_denominator, gs.ones_like(numerator), denominator
            )
        else:
            numerator = gs.where(
                null_denominator, power * powered_eigvalues[..., :, None], numerator
            )
            denominator = gs.where(
                null_denominator, eigvalues[..., :, None], denominator
            )

        transp_eigvectors = Matrices.transpose(eigvectors)
        temp_result = Matrices.mul(transp_eigvectors, tangent_vec, eigvectors)

        return (eigvectors, transp_eigvectors, numerator, denominator, temp_result)

    @classmethod
    def differential_power(cls, power, tangent_vec, base_point):
        r"""Compute the differential of the matrix power function.

        Compute the differential of the power function on SPD(n)
        (:math:`A^p=\exp(p \log(A))`) at base_point applied to tangent_vec.

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
        (
            eigvectors,
            transp_eigvectors,
            numerator,
            denominator,
            temp_result,
        ) = cls.aux_differential_power(power, tangent_vec, base_point)
        power_operator = numerator / denominator
        result = power_operator * temp_result
        result = Matrices.mul(eigvectors, result, transp_eigvectors)
        return result

    @classmethod
    def inverse_differential_power(cls, power, tangent_vec, base_point):
        r"""Compute the inverse of the differential of the matrix power.

        Compute the inverse of the differential of the power
        function on SPD matrices (:math:`A^p=\exp(p \log(A))`) at base_point
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
        (
            eigvectors,
            transp_eigvectors,
            numerator,
            denominator,
            temp_result,
        ) = cls.aux_differential_power(power, tangent_vec, base_point)
        power_operator = denominator / numerator
        result = power_operator * temp_result
        result = Matrices.mul(eigvectors, result, transp_eigvectors)
        return result

    @classmethod
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
        (
            eigvectors,
            transp_eigvectors,
            numerator,
            denominator,
            temp_result,
        ) = cls.aux_differential_power(0, tangent_vec, base_point)
        power_operator = numerator / denominator
        result = power_operator * temp_result
        result = Matrices.mul(eigvectors, result, transp_eigvectors)
        return result

    @classmethod
    def inverse_differential_log(cls, tangent_vec, base_point):
        """Compute the inverse of the differential of the matrix logarithm.

        Compute the inverse of the differential of the matrix
        logarithm on SPD matrices at base_point applied to tangent_vec.

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
        (
            eigvectors,
            transp_eigvectors,
            numerator,
            denominator,
            temp_result,
        ) = cls.aux_differential_power(0, tangent_vec, base_point)
        power_operator = denominator / numerator
        result = power_operator * temp_result
        result = Matrices.mul(eigvectors, result, transp_eigvectors)
        return result

    @classmethod
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
        (
            eigvectors,
            transp_eigvectors,
            numerator,
            denominator,
            temp_result,
        ) = cls.aux_differential_power(math.inf, tangent_vec, base_point)
        power_operator = numerator / denominator
        result = power_operator * temp_result
        result = Matrices.mul(eigvectors, result, transp_eigvectors)
        return result

    @classmethod
    def inverse_differential_exp(cls, tangent_vec, base_point):
        """Compute the inverse of the differential of the matrix exponential.

        Computes the inverse of the differential of the matrix
        exponential on SPD matrices at base_point applied to tangent_vec.

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
        (
            eigvectors,
            transp_eigvectors,
            numerator,
            denominator,
            temp_result,
        ) = cls.aux_differential_power(math.inf, tangent_vec, base_point)
        power_operator = denominator / numerator
        result = power_operator * temp_result
        result = Matrices.mul(eigvectors, result, transp_eigvectors)
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
        n = mat.shape[-1]
        dim_3_mat = gs.reshape(mat, [-1, n, n])
        logm = SymmetricMatrices.apply_func_to_eigvals(
            dim_3_mat, gs.log, check_positive=True
        )
        logm = gs.reshape(logm, mat.shape)
        return logm

    expm = SymmetricMatrices.expm
    powerm = SymmetricMatrices.powerm
    from_vector = SymmetricMatrices.__dict__["from_vector"]
    to_vector = SymmetricMatrices.__dict__["to_vector"]

    @classmethod
    def cholesky_factor(cls, mat):
        """Compute cholesky factor.

        Compute cholesky factor for a symmetric positive definite matrix.

        Parameters
        ----------
        mat : array_like, shape=[..., n, n]
            spd matrix.

        Returns
        -------
        cf : array_like, shape=[..., n, n]
            lower triangular matrix with positive diagonal elements.
        """
        return gs.linalg.cholesky(mat)

    @classmethod
    def differential_cholesky_factor(cls, tangent_vec, base_point):
        """Compute the differential of the cholesky factor map.

        Parameters
        ----------
        tangent_vec : array_like, shape=[..., n, n]
            Tangent vector at base point.
            symmetric matrix.

        base_point : array_like, shape=[..., n, n]
            Base point.
            spd matrix.

        Returns
        -------
        differential_cf : array-like, shape=[..., n, n]
            Differential of cholesky factor map
            lower triangular matrix.
        """
        cf = cls.cholesky_factor(base_point)
        differential_cf = PositiveLowerTriangularMatrices.inverse_differential_gram(
            tangent_vec, cf
        )
        return differential_cf


class SPDAffineMetric(RiemannianMetric):
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
            SPD matrices? A principled continuum of metrics" Proc. of GSI,
            2019. https://arxiv.org/abs/1906.01349
        """
        dim = int(n * (n + 1) / 2)
        super().__init__(
            dim=dim,
            shape=(n, n),
            signature=(dim, 0),
        )
        self.n = n
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
        inner_product : array-like, shape=[...]
        """
        aux_a = Matrices.mul(inv_base_point, tangent_vec_a)
        aux_b = Matrices.mul(inv_base_point, tangent_vec_b)

        # Use product instead of matrix product and trace to save time
        inner_product = Matrices.trace_product(aux_a, aux_b)

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
        spd_space = SPDMatrices

        if power_affine == 1:
            inv_base_point = GeneralLinear.inverse(base_point)
            inner_product = self._aux_inner_product(
                tangent_vec_a, tangent_vec_b, inv_base_point
            )
        else:
            modified_tangent_vec_a = spd_space.differential_power(
                power_affine, tangent_vec_a, base_point
            )
            modified_tangent_vec_b = spd_space.differential_power(
                power_affine, tangent_vec_b, base_point
            )
            power_inv_base_point = SymmetricMatrices.powerm(base_point, -power_affine)
            inner_product = self._aux_inner_product(
                modified_tangent_vec_a, modified_tangent_vec_b, power_inv_base_point
            )

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
        tangent_vec_at_id = Matrices.mul(
            inv_sqrt_base_point, tangent_vec, inv_sqrt_base_point
        )

        tangent_vec_at_id = Matrices.to_symmetric(tangent_vec_at_id)
        exp_from_id = SymmetricMatrices.expm(tangent_vec_at_id)

        exp = Matrices.mul(sqrt_base_point, exp_from_id, sqrt_base_point)
        return exp

    def exp(self, tangent_vec, base_point, **kwargs):
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
            powers = SymmetricMatrices.powerm(base_point, [1.0 / 2, -1.0 / 2])
            exp = self._aux_exp(tangent_vec, powers[0], powers[1])
        else:
            modified_tangent_vec = SPDMatrices.differential_power(
                power_affine, tangent_vec, base_point
            )
            power_sqrt_base_point = SymmetricMatrices.powerm(
                base_point, power_affine / 2
            )
            power_inv_sqrt_base_point = GeneralLinear.inverse(power_sqrt_base_point)
            exp = self._aux_exp(
                modified_tangent_vec, power_sqrt_base_point, power_inv_sqrt_base_point
            )
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
        point_near_id = Matrices.mul(inv_sqrt_base_point, point, inv_sqrt_base_point)
        point_near_id = Matrices.to_symmetric(point_near_id)

        log_at_id = SPDMatrices.logm(point_near_id)
        log = Matrices.mul(sqrt_base_point, log_at_id, sqrt_base_point)
        return log

    def log(self, point, base_point, **kwargs):
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
            powers = SymmetricMatrices.powerm(base_point, [1.0 / 2, -1.0 / 2])
            log = self._aux_log(point, powers[0], powers[1])
        else:
            power_point = SymmetricMatrices.powerm(point, power_affine)
            powers = SymmetricMatrices.powerm(
                base_point, [power_affine / 2, -power_affine / 2]
            )
            log = self._aux_log(power_point, powers[0], powers[1])
            log = SPDMatrices.inverse_differential_power(power_affine, log, base_point)
        return log

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None
    ):
        r"""Parallel transport of a tangent vector.

        Closed-form solution for the parallel transport of a tangent vector
        along the geodesic between two points `base_point` and `end_point`
        or alternatively defined by :math:`t \mapsto exp_{(base\_point)}(
        t*direction)`.
        Denoting `tangent_vec_a` by `S`, `base_point` by `A`, and `end_point`
        by `B` or `B = Exp_A(tangent_vec_b)` and :math:`E = (BA^{- 1})^{( 1
        / 2)}`. Then the parallel transport to `B` is:

        .. math::
            S' = ESE^T

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., n, n]
            Point on the manifold of SPD matrices. Point to transport from
        direction : array-like, shape=[..., n, n]
            Tangent vector at base point, initial speed of the geodesic along
            which the parallel transport is computed. Unused if `end_point` is given.
            Optional, default: None.
        end_point : array-like, shape=[..., n, n]
            Point on the manifold of SPD matrices. Point to transport to.
            Optional, default: None.

        Returns
        -------
        transported_tangent_vec: array-like, shape=[..., n, n]
            Transported tangent vector at exp_(base_point)(tangent_vec_b).
        """
        if end_point is None:
            end_point = self.exp(direction, base_point)
        # compute B^1/2(B^-1/2 A B^-1/2)B^-1/2 instead of sqrtm(AB^-1)
        sqrt_bp, inv_sqrt_bp = SymmetricMatrices.powerm(base_point, [1.0 / 2, -1.0 / 2])
        pdt = SymmetricMatrices.powerm(
            Matrices.mul(inv_sqrt_bp, end_point, inv_sqrt_bp), 1.0 / 2
        )
        congruence_mat = Matrices.mul(sqrt_bp, pdt, inv_sqrt_bp)
        return Matrices.congruent(tangent_vec, congruence_mat)

    def injectivity_radius(self, base_point):
        """Radius of the largest ball where the exponential is injective.

        Because of the negative curvature of this space, the injectivity radius
        is infinite everywhere.

        Parameters
        ----------
        base_point : array-like, shape=[..., n, n]
            Point on the manifold.

        Returns
        -------
        radius : float
            Injectivity radius.
        """
        return math.inf


class SPDBuresWassersteinMetric(RiemannianMetric):
    """Class for the Bures-Wasserstein metric on the SPD manifold.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.

    References
    ----------
    .. [BJL2017] Bhatia, Jain, Lim. "On the Bures-Wasserstein distance between
        positive definite matrices" Elsevier, Expositiones Mathematicae,
        vol. 37(2), 165-191, 2017. https://arxiv.org/pdf/1712.01504.pdf
    .. [MMP2018] Malago, Montrucchio, Pistone. "Wasserstein-Riemannian
        geometry of Gaussian densities"  Information Geometry, vol. 1, 137-179,
        2018. https://arxiv.org/pdf/1801.09269.pdf
    """

    def __init__(self, n):
        dim = int(n * (n + 1) / 2)
        super().__init__(
            dim=dim,
            signature=(dim, 0),
            shape=(n, n),
        )
        self.n = n

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        r"""Compute the Bures-Wasserstein inner-product.

        Compute the inner-product of tangent_vec_a :math:`A` and tangent_vec_b
        :math:`B` at point base_point :math:`S=PDP^\top` using the
        Bures-Wasserstein Riemannian metric:

        .. math::
            \frac{1}{2}\sum_{i,j}\frac{[P^\top AP]_{ij}[P^\top BP]_{ij}}{d_i+d_j}

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
        eigvals, eigvecs = gs.linalg.eigh(base_point)
        transp_eigvecs = Matrices.transpose(eigvecs)
        rotated_tangent_vec_a = Matrices.mul(transp_eigvecs, tangent_vec_a, eigvecs)
        rotated_tangent_vec_b = Matrices.mul(transp_eigvecs, tangent_vec_b, eigvecs)

        coefficients = 1 / (eigvals[..., :, None] + eigvals[..., None, :])
        result = (
            Matrices.frobenius_product(
                coefficients * rotated_tangent_vec_a, rotated_tangent_vec_b
            )
            / 2
        )

        return result

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the Bures-Wasserstein exponential map.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        exp : array-like, shape=[...,]
            Riemannian exponential.
        """
        eigvals, eigvecs = gs.linalg.eigh(base_point)
        transp_eigvecs = Matrices.transpose(eigvecs)
        rotated_tangent_vec = Matrices.mul(transp_eigvecs, tangent_vec, eigvecs)
        coefficients = 1 / (eigvals[..., :, None] + eigvals[..., None, :])
        rotated_sylvester = rotated_tangent_vec * coefficients
        rotated_hessian = gs.einsum("...ij,...j->...ij", rotated_sylvester, eigvals)
        rotated_hessian = Matrices.mul(rotated_hessian, rotated_sylvester)
        hessian = Matrices.mul(eigvecs, rotated_hessian, transp_eigvecs)

        return base_point + tangent_vec + hessian

    def log(self, point, base_point, **kwargs):
        """Compute the Bures-Wasserstein logarithm map.

        Compute the Riemannian logarithm at point base_point,
        of point wrt the Bures-Wasserstein metric.
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
        # compute B^1/2(B^-1/2 A B^-1/2)B^-1/2 instead of sqrtm(AB^-1)
        sqrt_bp, inv_sqrt_bp = SymmetricMatrices.powerm(base_point, [0.5, -0.5])
        pdt = SymmetricMatrices.powerm(Matrices.mul(sqrt_bp, point, sqrt_bp), 0.5)
        sqrt_product = Matrices.mul(sqrt_bp, pdt, inv_sqrt_bp)
        transp_sqrt_product = Matrices.transpose(sqrt_product)
        return sqrt_product + transp_sqrt_product - 2 * base_point

    def squared_dist(self, point_a, point_b, **kwargs):
        """Compute the Bures-Wasserstein squared distance.

        Compute the Riemannian squared distance between point_a and point_b.

        Parameters
        ----------
        point_a : array-like, shape=[..., n, n]
            Point.
        point_b : array-like, shape=[..., n, n]
            Point.

        Returns
        -------
        squared_dist : array-like, shape=[...]
            Riemannian squared distance.
        """
        product = gs.matmul(point_a, point_b)
        sqrt_product = gs.linalg.sqrtm(product)
        trace_a = gs.trace(point_a)
        trace_b = gs.trace(point_b)
        trace_prod = gs.trace(sqrt_product)

        return trace_a + trace_b - 2 * trace_prod

    def parallel_transport(
        self,
        tangent_vec_a,
        base_point,
        tangent_vec_b=None,
        end_point=None,
        n_steps=10,
        step="rk4",
    ):
        r"""Compute the parallel transport of a tangent vec along a geodesic.

        Approximation of the solution of the parallel transport of a tangent
        vector a along the geodesic defined by :math:`t \mapsto exp_{(
        base\_point)}(t* tangent\_vec\_b)`. The parallel transport equation is
        formulated in this case in [TP2021]_.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at `base_point` to transport.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector ar `base_point`, initial velocity of the geodesic to
            transport along.
        base_point : array-like, shape=[..., n, n]
            Initial point of the geodesic.
        end_point : array-like, shape=[..., n, n]
            Point to transport to.
            Optional, default: None.
        n_steps : int
            Number of steps to use to approximate the solution of the
            ordinary differential equation.
            Optional, default: 100
        step : str, {'euler', 'rk2', 'rk4'}
            Scheme to use in the integration scheme.
            Optional, default: 'rk4'.

        Returns
        -------
        transported :  array-like, shape=[..., n, n]
            Transported tangent vector at `exp_(base_point)(tangent_vec_b)`.

        References
        ----------
        .. [TP2021] Yann Thanwerdas, Xavier Pennec. O(n)-invariant Riemannian
            metrics on SPD matrices. 2021. ⟨hal-03338601v2⟩

        See Also
        --------
        Integration module: geomstats.integrator
        """
        if end_point is None:
            end_point = self.exp(tangent_vec_b, base_point)

        horizontal_lift_a = gs.linalg.solve_sylvester(
            base_point, base_point, tangent_vec_a
        )

        square_root_bp, inverse_square_root_bp = SymmetricMatrices.powerm(
            base_point, [0.5, -0.5]
        )
        end_point_lift = Matrices.mul(square_root_bp, end_point, square_root_bp)
        square_root_lift = SymmetricMatrices.powerm(end_point_lift, 0.5)

        horizontal_velocity = gs.matmul(inverse_square_root_bp, square_root_lift)
        partial_horizontal_velocity = Matrices.mul(horizontal_velocity, square_root_bp)
        partial_horizontal_velocity += Matrices.transpose(partial_horizontal_velocity)

        def force(state, time):
            horizontal_geodesic_t = (
                1 - time
            ) * square_root_bp + time * horizontal_velocity
            geodesic_t = (
                (1 - time) ** 2 * base_point
                + time * (1 - time) * partial_horizontal_velocity
                + time**2 * end_point
            )

            align = Matrices.mul(
                horizontal_geodesic_t,
                Matrices.transpose(horizontal_velocity - square_root_bp),
                state,
            )
            right = align + Matrices.transpose(align)
            return gs.linalg.solve_sylvester(geodesic_t, geodesic_t, -right)

        flow = integrate(force, horizontal_lift_a, n_steps=n_steps, step=step)
        final_align = Matrices.mul(end_point, flow[-1])
        return final_align + Matrices.transpose(final_align)

    def injectivity_radius(self, base_point):
        """Compute the upper bound of the injectivity domain.

        This is the smallest eigen value of the base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., n, n]
            Point on the manifold.

        Returns
        -------
        radius : float
            Injectivity radius.
        """
        eigen_values = gs.linalg.eigvalsh(base_point)
        return eigen_values[..., 0] ** 0.5


class SPDEuclideanMetric(RiemannianMetric):
    """Class for the Euclidean metric on the SPD manifold."""

    def __init__(self, n, power_euclidean=1):
        dim = int(n * (n + 1) / 2)
        super().__init__(
            dim=dim,
            signature=(dim, 0),
            shape=(n, n),
        )
        self.n = n
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
        spd_space = SPDMatrices

        if power_euclidean == 1:
            inner_product = Matrices.frobenius_product(tangent_vec_a, tangent_vec_b)
        else:
            modified_tangent_vec_a = spd_space.differential_power(
                power_euclidean, tangent_vec_a, base_point
            )
            modified_tangent_vec_b = spd_space.differential_power(
                power_euclidean, tangent_vec_b, base_point
            )

            inner_product = Matrices.frobenius_product(
                modified_tangent_vec_a, modified_tangent_vec_b
            ) / (power_euclidean**2)
        return inner_product

    @staticmethod
    @geomstats.vectorization.decorator(["matrix", "matrix"])
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
        invsqrt_base_point = SymmetricMatrices.powerm(base_point, -0.5)

        reduced_vec = gs.matmul(invsqrt_base_point, tangent_vec)
        reduced_vec = gs.matmul(reduced_vec, invsqrt_base_point)
        eigvals = gs.linalg.eigvalsh(reduced_vec)
        min_eig = gs.amin(eigvals, axis=1)
        max_eig = gs.amax(eigvals, axis=1)
        inf_value = gs.where(max_eig <= 0.0, gs.array(-math.inf), -1.0 / max_eig)
        inf_value = gs.to_ndarray(inf_value, to_ndim=2)
        sup_value = gs.where(min_eig >= 0.0, gs.array(-math.inf), -1.0 / min_eig)
        sup_value = gs.to_ndarray(sup_value, to_ndim=2)
        domain = gs.concatenate((inf_value, sup_value), axis=1)

        return domain

    def injectivity_radius(self, base_point):
        """Compute the upper bound of the injectivity domain.

        This is the smallest eigen value of the base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., n, n]
            Point on the manifold.

        Returns
        -------
        radius : float
            Injectivity radius.
        """
        eigen_values = gs.linalg.eigvalsh(base_point)
        return eigen_values[..., 0]

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the Euclidean exponential map.

        Compute the Euclidean exponential at point base_point
        of tangent vector tangent_vec.
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
            Euclidean exponential.
        """
        power_euclidean = self.power_euclidean

        if power_euclidean == 1:
            exp = tangent_vec + base_point
        else:
            exp = SymmetricMatrices.powerm(
                SymmetricMatrices.powerm(base_point, power_euclidean)
                + SPDMatrices.differential_power(
                    power_euclidean, tangent_vec, base_point
                ),
                1 / power_euclidean,
            )
        return exp

    def log(self, point, base_point, **kwargs):
        """Compute the Euclidean logarithm map.

        Compute the Euclidean logarithm at point base_point, of point.
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
            Euclidean logarithm.
        """
        power_euclidean = self.power_euclidean

        if power_euclidean == 1:
            log = point - base_point
        else:
            log = SPDMatrices.inverse_differential_power(
                power_euclidean,
                SymmetricMatrices.powerm(point, power_euclidean)
                - SymmetricMatrices.powerm(base_point, power_euclidean),
                base_point,
            )

        return log

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None
    ):
        r"""Compute the parallel transport of a tangent vector.

        Closed-form solution for the parallel transport of a tangent vector
        along the geodesic between two points `base_point` and `end_point`
        or alternatively defined by :math:`t \mapsto exp_{(base\_point)}(
        t*direction)`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., n, n]
            Point on the manifold. Point to transport from.
        direction : array-like, shape=[..., n, n]
            Tangent vector at base point, along which the parallel transport
            is computed.
            Optional, default: None.
        end_point : array-like, shape=[..., n, n]
            Point on the manifold. Point to transport to.
            Optional, default: None.

        Returns
        -------
        transported_tangent_vec: array-like, shape=[..., n, n]
            Transported tangent vector at `exp_(base_point)(tangent_vec_b)`.
        """
        if self.power_euclidean == 1:
            return tangent_vec
        raise NotImplementedError("Parallel transport is only implemented for power 1")


class SPDLogEuclideanMetric(RiemannianMetric):
    """Class for the Log-Euclidean metric on the SPD manifold.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n):
        dim = int(n * (n + 1) / 2)
        super().__init__(
            dim=dim,
            signature=(dim, 0),
            shape=(n, n),
        )
        self.n = n

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
        spd_space = SPDMatrices

        modified_tangent_vec_a = spd_space.differential_log(tangent_vec_a, base_point)
        modified_tangent_vec_b = spd_space.differential_log(tangent_vec_b, base_point)
        product = Matrices.trace_product(modified_tangent_vec_a, modified_tangent_vec_b)
        return product

    def exp(self, tangent_vec, base_point, **kwargs):
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
        log_base_point = SPDMatrices.logm(base_point)
        dlog_tangent_vec = SPDMatrices.differential_log(tangent_vec, base_point)
        exp = SymmetricMatrices.expm(log_base_point + dlog_tangent_vec)

        return exp

    def log(self, point, base_point, **kwargs):
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
        log = SPDMatrices.differential_exp(log_point - log_base_point, log_base_point)

        return log

    def injectivity_radius(self, base_point):
        """Radius of the largest ball where the exponential is injective.

        Because of this space is flat, the injectivity radius is infinite
        everywhere.

        Parameters
        ----------
        base_point : array-like, shape=[..., n, n]
            Point on the manifold.

        Returns
        -------
        radius : float
            Injectivity radius.
        """
        return math.inf

    def dist(self, point_a, point_b):
        """Compute log euclidean distance.

        Parameters
        ----------
        point_a : array-like, shape=[..., dim]
            Point.
        point_b : array-like, shape=[..., dim]
            Point.

        Returns
        -------
        dist : array-like, shape=[...,]
            Distance.
        """
        log_a = SPDMatrices.logm(point_a)
        log_b = SPDMatrices.logm(point_b)
        return MatricesMetric.norm(log_a - log_b)
