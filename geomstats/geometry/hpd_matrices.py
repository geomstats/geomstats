"""The manifold of Hermitian positive definite (HPD) matrices.

Lead author: Yann Cabanes.
"""

import math

import geomstats.backend as gs
import geomstats.vectorization
from geomstats.geometry.base import ComplexOpenSet
from geomstats.geometry.complex_matrices import ComplexMatrices, ComplexMatricesMetric
from geomstats.geometry.complex_riemannian_metric import ComplexRiemannianMetric
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.hermitian_matrices import HermitianMatrices
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.positive_lower_triangular_matrices import (
    PositiveLowerTriangularMatrices,
)
from geomstats.integrator import integrate


class HPDMatrices(ComplexOpenSet):
    """Class for the manifold of Hermitian positive definite (HPD) matrices.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.

    References
    ----------
    .. [Cabanes2022] Yann Cabanes. Multidimensional complex stationary
        centered Gaussian autoregressive time series machine learning
        in Poincaré and Siegel disks: application for audio and radar
        clutter classification, PhD thesis, 2022
    .. [JV2016] B. Jeuris and R. Vandebril. The Kahler mean of Block-Toeplitz
        matrices with Toeplitz structured blocks, 2016.
        https://epubs.siam.org/doi/pdf/10.1137/15M102112X
    """

    def __init__(self, n, **kwargs):
        kwargs.setdefault("metric", HPDAffineMetric(n))
        super().__init__(dim=n**2, embedding_space=HermitianMatrices(n), **kwargs)
        self.n = n

    @staticmethod
    def belongs(point, atol=gs.atol):
        """Check if a matrix is Hermitian with positive eigenvalues.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point to be checked.
        atol : float
            Tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if mat is an HPD matrix.
        """
        return ComplexMatrices.is_hpd(point, atol)

    def projection(self, point, atol=gs.atol):
        """Project a matrix to the space of HPD matrices.

        First the Hermitian part of point is computed, then the eigenvalues
        are floored to gs.atol.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix to project.
        atol : float
            Tolerance.
            Optional, default: backend atol.

        Returns
        -------
        projected: array-like, shape=[..., n, n]
            HPD matrix.
        """
        herm = ComplexMatrices.to_hermitian(point)
        eigvals, eigvecs = gs.linalg.eigh(herm)
        regularized = gs.where(eigvals < atol, atol, eigvals)
        reconstruction = gs.einsum("...ij,...j->...ij", eigvecs, regularized)
        return Matrices.mul(reconstruction, ComplexMatrices.transconjugate(eigvecs))

    def random_point(self, n_samples=1, bound=0.1):
        """Sample in HPD(n) from the log-uniform distribution.

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
            Points sampled in HPD(n).
        """
        n = self.n
        size = (n_samples, n, n) if n_samples != 1 else (n, n)
        eye = gs.eye(n, dtype=gs.get_default_cdtype())
        samples = gs.stack([eye for i_sample in range(n_samples)], axis=0)
        samples = gs.reshape(samples, size)
        samples += bound * gs.random.rand(*size, dtype=gs.get_default_cdtype())
        samples = self.projection(samples)
        return samples

    def random_tangent_vec(self, base_point, n_samples=1):
        """Sample on the tangent space of HPD(n) from the uniform distribution.

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
            base_point = gs.eye(n, dtype=gs.get_default_cdtype())

        sqrt_base_point = gs.linalg.sqrtm(base_point)

        tangent_vec_at_id_aux = gs.random.rand(*size, dtype=gs.get_default_cdtype())
        tangent_vec_at_id_aux *= 2
        tangent_vec_at_id_aux -= 1 + 1j
        tangent_vec_at_id = tangent_vec_at_id_aux + ComplexMatrices.transconjugate(
            tangent_vec_at_id_aux
        )

        tangent_vec = Matrices.mul(sqrt_base_point, tangent_vec_at_id, sqrt_base_point)

        return tangent_vec

    @staticmethod
    def _aux_differential_power(power, tangent_vec, base_point):
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
        transconj_eigvectors : array-like, shape=[..., n, n]
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

        transconj_eigvectors = ComplexMatrices.transconjugate(eigvectors)
        temp_result = Matrices.mul(transconj_eigvectors, tangent_vec, eigvectors)

        numerator = gs.cast(numerator, dtype=temp_result.dtype)
        denominator = gs.cast(denominator, dtype=temp_result.dtype)

        return (eigvectors, transconj_eigvectors, numerator, denominator, temp_result)

    @classmethod
    def differential_power(cls, power, tangent_vec, base_point):
        r"""Compute the differential of the matrix power function.

        Compute the differential of the power function on HPD(n)
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
            transconj_eigvectors,
            numerator,
            denominator,
            temp_result,
        ) = cls._aux_differential_power(power, tangent_vec, base_point)
        power_operator = numerator / denominator
        result = power_operator * temp_result
        result = Matrices.mul(eigvectors, result, transconj_eigvectors)
        return result

    @classmethod
    def inverse_differential_power(cls, power, tangent_vec, base_point):
        r"""Compute the inverse of the differential of the matrix power.

        Compute the inverse of the differential of the power
        function on HPD matrices (:math:`A^p=\exp(p \log(A))`) at base_point
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
            transconj_eigvectors,
            numerator,
            denominator,
            temp_result,
        ) = cls._aux_differential_power(power, tangent_vec, base_point)
        power_operator = denominator / numerator
        result = power_operator * temp_result
        result = Matrices.mul(eigvectors, result, transconj_eigvectors)
        return result

    @classmethod
    def differential_log(cls, tangent_vec, base_point):
        """Compute the differential of the matrix logarithm.

        Compute the differential of the matrix logarithm on HPD
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
            transconj_eigvectors,
            numerator,
            denominator,
            temp_result,
        ) = cls._aux_differential_power(0, tangent_vec, base_point)
        power_operator = numerator / denominator
        result = power_operator * temp_result
        result = Matrices.mul(eigvectors, result, transconj_eigvectors)
        return result

    @classmethod
    def inverse_differential_log(cls, tangent_vec, base_point):
        """Compute the inverse of the differential of the matrix logarithm.

        Compute the inverse of the differential of the matrix
        logarithm on HPD matrices at base_point applied to tangent_vec.

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
            transconj_eigvectors,
            numerator,
            denominator,
            temp_result,
        ) = cls._aux_differential_power(0, tangent_vec, base_point)
        power_operator = denominator / numerator
        result = power_operator * temp_result
        result = Matrices.mul(eigvectors, result, transconj_eigvectors)
        return result

    @classmethod
    def differential_exp(cls, tangent_vec, base_point):
        """Compute the differential of the matrix exponential.

        Computes the differential of the matrix exponential on HPD
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
            transconj_eigvectors,
            numerator,
            denominator,
            temp_result,
        ) = cls._aux_differential_power(math.inf, tangent_vec, base_point)
        power_operator = numerator / denominator
        result = power_operator * temp_result
        result = Matrices.mul(eigvectors, result, transconj_eigvectors)
        return result

    @classmethod
    def inverse_differential_exp(cls, tangent_vec, base_point):
        """Compute the inverse of the differential of the matrix exponential.

        Computes the inverse of the differential of the matrix
        exponential on HPD matrices at base_point applied to tangent_vec.

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
            transconj_eigvectors,
            numerator,
            denominator,
            temp_result,
        ) = cls._aux_differential_power(math.inf, tangent_vec, base_point)
        power_operator = denominator / numerator
        result = power_operator * temp_result
        result = Matrices.mul(eigvectors, result, transconj_eigvectors)
        return result

    @classmethod
    def logm(cls, mat):
        """
        Compute the matrix log for a Hermitian matrix.

        Parameters
        ----------
        mat : array_like, shape=[..., n, n]
            Hermitian matrix.

        Returns
        -------
        log : array_like, shape=[..., n, n]
            Matrix logarithm of mat.
        """
        n = mat.shape[-1]
        dim_3_mat = gs.reshape(mat, [-1, n, n])
        logm = HermitianMatrices.apply_func_to_eigvals(
            dim_3_mat, gs.log, check_positive=True
        )
        logm = gs.reshape(logm, mat.shape)
        return logm

    expm = HermitianMatrices.expm
    powerm = HermitianMatrices.powerm
    from_vector = HermitianMatrices.__dict__["from_vector"]
    to_vector = HermitianMatrices.__dict__["to_vector"]

    @classmethod
    def cholesky_factor(cls, mat):
        """Compute cholesky factor.

        Compute cholesky factor for a Hermitian positive definite matrix.

        Parameters
        ----------
        mat : array_like, shape=[..., n, n]
            HPD matrix.

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
            Hermitian matrix.

        base_point : array_like, shape=[..., n, n]
            Base point.
            HPD matrix.

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


class HPDAffineMetric(ComplexRiemannianMetric):
    """Class for the affine-invariant metric on the HPD manifold.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    power_affine : int
        Power transformation of the classical HPD metric.
        Optional, default: 1.

    References
    ----------
    .. [Cabanes2022] Yann Cabanes. Multidimensional complex stationary
        centered Gaussian autoregressive time series machine learning
        in Poincaré and Siegel disks: application for audio and radar
        clutter classification, PhD thesis, 2022
    .. [JV2016] B. Jeuris and R. Vandebril. The Kahler mean of Block-Toeplitz
        matrices with Toeplitz structured blocks, 2016.
        https://epubs.siam.org/doi/pdf/10.1137/15M102112X
    """

    def __init__(self, n, power_affine=1, **kwargs):
        if "scale" in kwargs:
            raise TypeError(
                "Argument scale is no longer in use: instantiate scaled "
                "metrics as `scale * RiemannianMetric`. Note that the "
                "metric is scaled, not the distance."
            )
        dim = n**2
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
        hpd_space = HPDMatrices

        if power_affine == 1:
            inv_base_point = GeneralLinear.inverse(base_point)
            inner_product = self._aux_inner_product(
                tangent_vec_a, tangent_vec_b, inv_base_point
            )
        else:
            modified_tangent_vec_a = hpd_space.differential_power(
                power_affine, tangent_vec_a, base_point
            )
            modified_tangent_vec_b = hpd_space.differential_power(
                power_affine, tangent_vec_b, base_point
            )
            power_inv_base_point = HermitianMatrices.powerm(base_point, -power_affine)
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

        tangent_vec_at_id = ComplexMatrices.to_hermitian(tangent_vec_at_id)
        exp_from_id = HermitianMatrices.expm(tangent_vec_at_id)

        exp = Matrices.mul(sqrt_base_point, exp_from_id, sqrt_base_point)
        return exp

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the affine-invariant exponential map.

        Compute the Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the metric defined in inner_product.
        This gives a Hermitian positive definite matrix.

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
            powers = HermitianMatrices.powerm(base_point, [1.0 / 2, -1.0 / 2])
            exp = self._aux_exp(tangent_vec, powers[0], powers[1])
        else:
            modified_tangent_vec = HPDMatrices.differential_power(
                power_affine, tangent_vec, base_point
            )
            power_sqrt_base_point = HermitianMatrices.powerm(
                base_point, power_affine / 2
            )
            power_inv_sqrt_base_point = GeneralLinear.inverse(power_sqrt_base_point)
            exp = self._aux_exp(
                modified_tangent_vec, power_sqrt_base_point, power_inv_sqrt_base_point
            )
            exp = HermitianMatrices.powerm(exp, 1 / power_affine)

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
        point_near_id = ComplexMatrices.to_hermitian(point_near_id)

        log_at_id = HPDMatrices.logm(point_near_id)
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
            powers = HermitianMatrices.powerm(base_point, [1.0 / 2, -1.0 / 2])
            log = self._aux_log(point, powers[0], powers[1])
        else:
            power_point = HermitianMatrices.powerm(point, power_affine)
            powers = HermitianMatrices.powerm(
                base_point, [power_affine / 2, -power_affine / 2]
            )
            log = self._aux_log(power_point, powers[0], powers[1])
            log = HPDMatrices.inverse_differential_power(power_affine, log, base_point)
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
            Point on the manifold of HPD matrices. Point to transport from
        direction : array-like, shape=[..., n, n]
            Tangent vector at base point, initial speed of the geodesic along
            which the parallel transport is computed. Unused if `end_point` is given.
            Optional, default: None.
        end_point : array-like, shape=[..., n, n]
            Point on the manifold of HPD matrices. Point to transport to.
            Optional, default: None.

        Returns
        -------
        transported_tangent_vec: array-like, shape=[..., n, n]
            Transported tangent vector at exp_(base_point)(tangent_vec_b).
        """
        if end_point is None:
            end_point = self.exp(direction, base_point)
        sqrt_bp, inv_sqrt_bp = HermitianMatrices.powerm(base_point, [1.0 / 2, -1.0 / 2])
        pdt = HermitianMatrices.powerm(
            Matrices.mul(inv_sqrt_bp, end_point, inv_sqrt_bp), 1.0 / 2
        )
        congruence_mat = Matrices.mul(sqrt_bp, pdt, inv_sqrt_bp)
        return ComplexMatrices.congruent(tangent_vec, congruence_mat)

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


class HPDBuresWassersteinMetric(ComplexRiemannianMetric):
    """Class for the Bures-Wasserstein metric on the HPD manifold.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n):
        dim = n**2
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
        transconj_eigvecs = ComplexMatrices.transconjugate(eigvecs)
        rotated_tangent_vec_a = Matrices.mul(transconj_eigvecs, tangent_vec_a, eigvecs)
        rotated_tangent_vec_b = Matrices.mul(transconj_eigvecs, tangent_vec_b, eigvecs)

        coefficients = 1 / (eigvals[..., :, None] + eigvals[..., None, :])
        result = (
            ComplexMatrices.frobenius_product(
                gs.cast(coefficients, dtype=rotated_tangent_vec_a.dtype)
                * rotated_tangent_vec_a,
                rotated_tangent_vec_b,
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
        transconj_eigvecs = ComplexMatrices.transconjugate(eigvecs)
        rotated_tangent_vec = Matrices.mul(transconj_eigvecs, tangent_vec, eigvecs)
        coefficients = 1 / (eigvals[..., :, None] + eigvals[..., None, :])
        rotated_sylvester = rotated_tangent_vec * gs.cast(
            coefficients, dtype=rotated_tangent_vec.dtype
        )
        rotated_hessian = gs.einsum("...ij,...j->...ij", rotated_sylvester, eigvals)
        rotated_hessian = Matrices.mul(rotated_hessian, rotated_sylvester)
        hessian = Matrices.mul(eigvecs, rotated_hessian, transconj_eigvecs)

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
        sqrt_bp, inv_sqrt_bp = HermitianMatrices.powerm(base_point, [0.5, -0.5])
        pdt = HermitianMatrices.powerm(Matrices.mul(sqrt_bp, point, sqrt_bp), 0.5)
        sqrt_product = Matrices.mul(sqrt_bp, pdt, inv_sqrt_bp)
        transconj_sqrt_product = ComplexMatrices.transconjugate(sqrt_product)
        return sqrt_product + transconj_sqrt_product - 2 * base_point

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
        return gs.real(trace_a + trace_b - 2 * trace_prod)

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

        See Also
        --------
        Integration module: geomstats.integrator
        """
        if end_point is None:
            end_point = self.exp(tangent_vec_b, base_point)

        horizontal_lift_a = gs.linalg.solve_sylvester(
            base_point, base_point, tangent_vec_a
        )

        square_root_bp, inverse_square_root_bp = HermitianMatrices.powerm(
            base_point, [0.5, -0.5]
        )
        end_point_lift = Matrices.mul(square_root_bp, end_point, square_root_bp)
        square_root_lift = HermitianMatrices.powerm(end_point_lift, 0.5)

        horizontal_velocity = gs.matmul(inverse_square_root_bp, square_root_lift)
        partial_horizontal_velocity = Matrices.mul(horizontal_velocity, square_root_bp)
        partial_horizontal_velocity += ComplexMatrices.transconjugate(
            partial_horizontal_velocity
        )

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
                ComplexMatrices.transconjugate(horizontal_velocity - square_root_bp),
                state,
            )
            right = align + ComplexMatrices.transconjugate(align)
            return gs.linalg.solve_sylvester(geodesic_t, geodesic_t, -right)

        flow = integrate(force, horizontal_lift_a, n_steps=n_steps, step=step)
        final_align = Matrices.mul(end_point, flow[-1])
        return final_align + ComplexMatrices.transconjugate(final_align)

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


class HPDEuclideanMetric(ComplexRiemannianMetric):
    """Class for the Euclidean metric on the HPD manifold."""

    def __init__(self, n, power_euclidean=1):
        dim = n**2
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
        hpd_space = HPDMatrices

        if power_euclidean == 1:
            inner_product = ComplexMatrices.frobenius_product(
                tangent_vec_a, tangent_vec_b
            )
        else:
            modified_tangent_vec_a = hpd_space.differential_power(
                power_euclidean, tangent_vec_a, base_point
            )
            modified_tangent_vec_b = hpd_space.differential_power(
                power_euclidean, tangent_vec_b, base_point
            )

            inner_product = ComplexMatrices.frobenius_product(
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
        invsqrt_base_point = HermitianMatrices.powerm(base_point, -0.5)

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
        This gives a Hermitian positive definite matrix.

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
            exp = HermitianMatrices.powerm(
                HermitianMatrices.powerm(base_point, power_euclidean)
                + HPDMatrices.differential_power(
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
            log = HPDMatrices.inverse_differential_power(
                power_euclidean,
                HermitianMatrices.powerm(point, power_euclidean)
                - HermitianMatrices.powerm(base_point, power_euclidean),
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
            return gs.copy(tangent_vec)
        raise NotImplementedError("Parallel transport is only implemented for power 1")


class HPDLogEuclideanMetric(ComplexRiemannianMetric):
    """Class for the Log-Euclidean metric on the HPD manifold.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n):
        dim = n**2
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
        hpd_space = HPDMatrices

        modified_tangent_vec_a = hpd_space.differential_log(tangent_vec_a, base_point)
        modified_tangent_vec_b = hpd_space.differential_log(tangent_vec_b, base_point)
        product = Matrices.trace_product(modified_tangent_vec_a, modified_tangent_vec_b)
        return product

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the Log-Euclidean exponential map.

        Compute the Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the Log-Euclidean metric.
        This gives a Hermitian positive definite matrix.

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
        log_base_point = HPDMatrices.logm(base_point)
        dlog_tangent_vec = HPDMatrices.differential_log(tangent_vec, base_point)
        exp = HermitianMatrices.expm(log_base_point + dlog_tangent_vec)

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
        log_base_point = HPDMatrices.logm(base_point)
        log_point = HPDMatrices.logm(point)
        log = HPDMatrices.differential_exp(log_point - log_base_point, log_base_point)

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
        log_a = HPDMatrices.logm(point_a)
        log_b = HPDMatrices.logm(point_b)
        return ComplexMatricesMetric.norm(log_a - log_b)
