"""The manifold of symmetric positive definite (SPD) matrices.

Lead authors: Yann Thanwerdas and Olivier Bisson.
"""

import math

import geomstats.backend as gs
from geomstats.algebra_utils import columnwise_scaling
from geomstats.geometry.base import VectorSpaceOpenSet
from geomstats.geometry.complex_matrices import ComplexMatrices
from geomstats.geometry.diffeo import Diffeo
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.hermitian_matrices import apply_func_to_eigvalsh, expmh, powermh
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.geometry.positive_lower_triangular_matrices import (
    InvariantPositiveLowerTriangularMatricesMetric,
    PositiveLowerTriangularMatrices,
)
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.scalar_product_metric import ScalarProductMetric
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.integrator import integrate
from geomstats.vectorization import repeat_out


def logmh(mat):
    """Compute the matrix log for a Hermitian matrix."""
    n = mat.shape[-1]
    dim_3_mat = gs.reshape(mat, [-1, n, n])
    logm = apply_func_to_eigvalsh(dim_3_mat, gs.log, check_positive=True)
    return gs.reshape(logm, mat.shape)


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
        denominator = gs.where(null_denominator, eigvalues[..., :, None], denominator)
    elif power == math.inf:
        numerator = gs.where(
            null_denominator, powered_eigvalues[..., :, None], numerator
        )
        denominator = gs.where(null_denominator, gs.ones_like(numerator), denominator)
    else:
        numerator = gs.where(
            null_denominator, power * powered_eigvalues[..., :, None], numerator
        )
        denominator = gs.where(null_denominator, eigvalues[..., :, None], denominator)

    if gs.is_complex(base_point):
        transp_eigvectors = ComplexMatrices.transconjugate(eigvectors)
    else:
        transp_eigvectors = Matrices.transpose(eigvectors)

    temp_result = Matrices.mul(transp_eigvectors, tangent_vec, eigvectors)

    if gs.is_complex(base_point):
        numerator = gs.cast(numerator, dtype=temp_result.dtype)
        denominator = gs.cast(denominator, dtype=temp_result.dtype)

    return (eigvectors, transp_eigvectors, numerator, denominator, temp_result)


def generalized_eigenvalues(point_a, point_b):
    """Compute the generalized eigenvalues of SPD matrix pair.

    Steps (check section 7.2 of [GKC2023]_):
    1. compute eigendecomposition of point_b
    2. get matrix turning point_b into identity by congruence
    3. apply congruence to point_a and get generalized eigenvalues

    Parameters
    ----------
    point_a : array_like, shape=[..., n, n]
        Symmetric positive definite matrix.
    point_b : array_like, shape=[..., n, n]
        Symmetric positive definite matrix.

    Returns
    -------
    generalized_eigenvalues : array-like, shape=[...]

    References
    ----------
    .. [GKC2023] Benyamin Ghojogh, Fakhri Karray, and Mark Crowley.
    “Eigenvalue and Generalized Eigenvalue Problems: Tutorial.”
    arXiv, May 20, 2023. https://doi.org/10.48550/arXiv.1903.11240.
    """
    eigvals_b, eigvecs_b = gs.linalg.eigh(point_b)

    inv_sqrt_eigvals_b = gs.sqrt(1.0 / eigvals_b)
    scaled_b_eigvecs = columnwise_scaling(inv_sqrt_eigvals_b, eigvecs_b)

    point_a_scaled = Matrices.mul(
        Matrices.transpose(scaled_b_eigvecs), point_a, scaled_b_eigvecs
    )
    return gs.linalg.eigvalsh(point_a_scaled)


class SymMatrixLog(Diffeo):
    """Matrix logarithm diffeomorphism.

    A diffeomorphism from the space of symmetric positive-definite matrices to
    the space of symmetric matrices.
    """

    @classmethod
    def __call__(cls, base_point):
        """Compute the matrix log for a symmetric matrix.

        Parameters
        ----------
        base_point : array_like, shape=[..., n, n]
            Symmetric matrix.

        Returns
        -------
        log : array_like, shape=[..., n, n]
            Matrix logarithm of base_point.
        """
        return logmh(base_point)

    @classmethod
    def inverse(cls, image_point):
        """Compute the matrix exponential for a symmetric matrix.

        Parameters
        ----------
        image_point : array_like, shape=[..., n, n]
            Symmetric matrix.

        Returns
        -------
        exponential : array_like, shape=[..., n, n]
            Exponential of image_point.
        """
        return expmh(image_point)

    @classmethod
    def tangent(cls, tangent_vec, base_point=None, image_point=None):
        """Compute the differential of the matrix logarithm.

        Compute the differential of the matrix logarithm on SPD
        matrices at base_point applied to tangent_vec.

        Parameters
        ----------
        tangent_vec : array_like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array_like, shape=[..., n, n]
            Base point.
        image_point : array_like, shape=[..., n, n]
            Image base point.

        Returns
        -------
        differential_log : array-like, shape=[..., n, n]
            Differential of the matrix logarithm.
        """
        if base_point is not None:
            (
                eigvectors,
                transp_eigvectors,
                numerator,
                denominator,
                temp_result,
            ) = _aux_differential_power(0, tangent_vec, base_point)
            power_operator = numerator / denominator
        else:
            (
                eigvectors,
                transp_eigvectors,
                numerator,
                denominator,
                temp_result,
            ) = _aux_differential_power(math.inf, tangent_vec, image_point)
            power_operator = denominator / numerator

        result = power_operator * temp_result
        return Matrices.mul(eigvectors, result, transp_eigvectors)

    @classmethod
    def inverse_tangent(cls, image_tangent_vec, image_point=None, base_point=None):
        """Compute the differential of the matrix exponential.

        Computes the differential of the matrix exponential on SPD
        matrices at base_point applied to tangent_vec.

        Parameters
        ----------
        image_tangent_vec : array_like, shape=[..., n, n]
            Image tangent vector at image point.
        image_point : array_like, shape=[..., n, n]
            Image point.
        base_point : array_like, shape=[..., n, n]
            Base point.

        Returns
        -------
        differential_exp : array-like, shape=[..., n, n]
            Differential of the matrix exponential.
        """
        if image_point is not None:
            (
                eigvectors,
                transp_eigvectors,
                numerator,
                denominator,
                temp_result,
            ) = _aux_differential_power(math.inf, image_tangent_vec, image_point)
            power_operator = numerator / denominator
        else:
            (
                eigvectors,
                transp_eigvectors,
                numerator,
                denominator,
                temp_result,
            ) = _aux_differential_power(0, image_tangent_vec, base_point)
            power_operator = denominator / numerator

        result = power_operator * temp_result
        return Matrices.mul(eigvectors, result, transp_eigvectors)


class MatrixPower(Diffeo):
    """Matrix power diffeomorphism.

    A diffeomorphism from the space of symmetric positive-definite matrices to
    itself.
    """

    def __init__(self, power):
        self.power = power

    def __call__(self, base_point):
        """Compute the matrix power.

        Parameters
        ----------
        base_point : array_like, shape=[..., n, n]
            Symmetric matrix with non-negative eigenvalues.

        Returns
        -------
        powerm : array_like or list of arrays, shape=[..., n, n]
            Matrix power of mat.
        """
        return powermh(base_point, self.power)

    def inverse(self, image_point):
        """Compute the inverse matrix power.

        Parameters
        ----------
        image_point : array_like, shape=[..., n, n]
            Symmetric matrix with non-negative eigenvalues.

        Returns
        -------
        powerm : array_like or list of arrays, shape=[..., n, n]
            Matrix power of mat.
        """
        return powermh(image_point, 1 / self.power)

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        r"""Compute the differential of the matrix power function.

        Compute the differential of the power function on SPD(n),
        :math:`A^p=\exp(p \log(A))`, at base_point applied to tangent_vec.

        Parameters
        ----------
        tangent_vec : array_like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array_like, shape=[..., n, n]
            Base point.
        image_point : array_like, shape=[..., n, n]
            Image base point.

        Returns
        -------
        differential_power : array-like, shape=[..., n, n]
            Differential of the power function.
        """
        if base_point is None:
            base_point = self.inverse(image_point)
        (
            eigvectors,
            transp_eigvectors,
            numerator,
            denominator,
            temp_result,
        ) = _aux_differential_power(self.power, tangent_vec, base_point)

        power_operator = numerator / denominator
        result = power_operator * temp_result
        return Matrices.mul(eigvectors, result, transp_eigvectors)

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
        r"""Compute the inverse of the differential of the matrix power.

        Compute the inverse of the differential of the power
        function on SPD matrices, :math:`A^p=\exp(p \log(A))`, at base_point
        applied to tangent_vec.

        Parameters
        ----------
        image_tangent_vec : array_like, shape=[..., n, n]
            Image tangent vector at image point.
        image_base_point : array_like, shape=[..., n, n]
            Image point.
        base_point : array_like, shape=[..., n, n]
            Base point.

        Returns
        -------
        inverse_differential_power : array-like, shape=[..., n, n]
            Inverse of the differential of the power function.
        """
        if base_point is None:
            base_point = self.inverse(image_point)
        (
            eigvectors,
            transp_eigvectors,
            numerator,
            denominator,
            temp_result,
        ) = _aux_differential_power(self.power, image_tangent_vec, base_point)
        power_operator = denominator / numerator
        result = power_operator * temp_result
        return Matrices.mul(eigvectors, result, transp_eigvectors)


class CholeskyMap(Diffeo):
    """Cholesky map.

    A diffeomorphism from the space of symmetric positive-definite matrices to
    the space of positive lower triangular matrices.
    """

    @classmethod
    def __call__(cls, base_point):
        """Compute cholesky factor.

        Compute cholesky factor for a symmetric positive definite matrix.

        Parameters
        ----------
        base_point : array_like, shape=[..., n, n]
            spd matrix.

        Returns
        -------
        cf : array_like, shape=[..., n, n]
            lower triangular matrix with positive diagonal elements.
        """
        return gs.linalg.cholesky(base_point)

    @staticmethod
    def inverse(image_point):
        """Compute gram matrix of rows.

        Gram_matrix is mapping from point to point.point^{T}.
        This is diffeomorphism between cholesky space and spd manifold.

        Parameters
        ----------
        image_point : array-like, shape=[..., n, n]
            element in cholesky space.

        Returns
        -------
        projected: array-like, shape=[..., n, n]
            SPD matrix.
        """
        return gs.einsum("...ij,...kj->...ik", image_point, image_point)

    @classmethod
    def tangent(cls, tangent_vec, base_point=None, image_point=None):
        """Compute the differential of the cholesky factor map.

        Parameters
        ----------
        tangent_vec : array_like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array_like, shape=[..., n, n]
            Base point.
        image_point : array_like, shape=[..., n, n]
            Image base point.

        Returns
        -------
        differential_cf : array-like, shape=[..., n, n]
            Differential of cholesky factor map
            lower triangular matrix.
        """
        if image_point is None:
            image_point = cls.__call__(base_point)

        inv_base_point = gs.linalg.inv(image_point)
        inv_transpose_base_point = Matrices.transpose(inv_base_point)
        aux = Matrices.to_lower_triangular_diagonal_scaled(
            Matrices.mul(inv_base_point, tangent_vec, inv_transpose_base_point)
        )
        return Matrices.mul(image_point, aux)

    @classmethod
    def inverse_tangent(cls, image_tangent_vec, image_point=None, base_point=None):
        """Compute differential of gram.

        Parameters
        ----------
        image_tangent_vec : array_like, shape=[..., n, n]
            Tangent vector at base point.
        image_point : array_like, shape=[..., n, n]
            Base point.
        base_point : array_like, shape=[..., n, n]
            Base point.

        Returns
        -------
        differential_gram : array-like, shape=[..., n, n]
            Differential of the gram.
        """
        if image_point is None:
            image_point = cls.__call__(base_point)

        mat1 = gs.einsum("...ij,...kj->...ik", image_tangent_vec, image_point)
        mat2 = gs.einsum("...ij,...kj->...ik", image_point, image_tangent_vec)
        return mat1 + mat2


class SPDMatrices(VectorSpaceOpenSet):
    """Class for the manifold of symmetric positive definite (SPD) matrices.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    equip : bool
        If True, equip space with default metric.
    """

    def __init__(self, n, equip=True):
        super().__init__(
            dim=int(n * (n + 1) / 2),
            embedding_space=SymmetricMatrices(n),
            equip=equip,
        )
        self.n = n

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return SPDAffineMetric

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
        return gs.logical_and(is_sym, is_pd)

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
        return GeneralLinear.exp(Matrices.to_symmetric(mat))

    def random_tangent_vec(self, base_point, n_samples=1):
        """Sample on the tangent space of SPD(n) from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        base_point : array-like, shape=[..., n, n]
            Base point of the tangent space.

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Points sampled in the tangent space at base_point.
        """
        n = self.n
        size = (n_samples, n, n) if n_samples != 1 else (n, n)

        sqrt_base_point = gs.linalg.sqrtm(base_point)

        tangent_vec_at_id_aux = 2 * gs.random.rand(*size) - 1
        tangent_vec_at_id = tangent_vec_at_id_aux + Matrices.transpose(
            tangent_vec_at_id_aux
        )

        return Matrices.mul(sqrt_base_point, tangent_vec_at_id, sqrt_base_point)


class SPDAffineMetric(RiemannianMetric):
    """Class for the affine-invariant metric on the SPD manifold.

    Parameters
    ----------
    power_affine : int
        Power transformation of the classical SPD metric.
        Optional, default: 1.

    References
    ----------
    .. [TP2019] Thanwerdas, Pennec. "Is affine-invariance well defined on
        SPD matrices? A principled continuum of metrics" Proc. of GSI,
        2019. https://arxiv.org/abs/1906.01349
    """

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
        inv_base_point = GeneralLinear.inverse(base_point)
        aux_a = Matrices.mul(inv_base_point, tangent_vec_a)
        aux_b = Matrices.mul(inv_base_point, tangent_vec_b)

        return Matrices.trace_product(aux_a, aux_b)

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
        sqrt_base_point, inv_sqrt_base_point = powermh(base_point, [1.0 / 2, -1.0 / 2])

        tangent_vec_at_id = Matrices.mul(
            inv_sqrt_base_point, tangent_vec, inv_sqrt_base_point
        )

        tangent_vec_at_id = Matrices.to_symmetric(tangent_vec_at_id)
        exp_from_id = expmh(tangent_vec_at_id)

        return Matrices.mul(sqrt_base_point, exp_from_id, sqrt_base_point)

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
        sqrt_base_point, inv_sqrt_base_point = powermh(base_point, [1.0 / 2, -1.0 / 2])
        point_near_id = Matrices.mul(inv_sqrt_base_point, point, inv_sqrt_base_point)
        point_near_id = Matrices.to_symmetric(point_near_id)

        log_at_id = logmh(point_near_id)
        return Matrices.mul(sqrt_base_point, log_at_id, sqrt_base_point)

    def squared_dist(self, point_a, point_b):
        """Compute the Affine Invariant squared distance.

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
        gen_eigvals = generalized_eigenvalues(point_a, point_b)
        return gs.sum(gs.log(gen_eigvals) ** 2, axis=-1)

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
        sqrt_bp, inv_sqrt_bp = powermh(base_point, [1.0 / 2, -1.0 / 2])
        pdt = powermh(Matrices.mul(inv_sqrt_bp, end_point, inv_sqrt_bp), 1.0 / 2)
        congruence_mat = Matrices.mul(sqrt_bp, pdt, inv_sqrt_bp)
        return Matrices.congruent(tangent_vec, congruence_mat)

    def injectivity_radius(self, base_point=None):
        """Radius of the largest ball where the exponential is injective.

        Because of the negative curvature of this space, the injectivity radius
        is infinite everywhere.

        Parameters
        ----------
        base_point : array-like, shape=[..., n, n]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        radius = gs.array(math.inf)
        return repeat_out(self._space.point_ndim, radius, base_point)


class SPDBuresWassersteinMetric(RiemannianMetric):
    """Class for the Bures-Wasserstein metric on the SPD manifold.

    References
    ----------
    .. [BJL2017] Bhatia, Jain, Lim. "On the Bures-Wasserstein distance between
        positive definite matrices" Elsevier, Expositiones Mathematicae,
        vol. 37(2), 165-191, 2017. https://arxiv.org/pdf/1712.01504.pdf
    .. [MMP2018] Malago, Montrucchio, Pistone. "Wasserstein-Riemannian
        geometry of Gaussian densities"  Information Geometry, vol. 1, 137-179,
        2018. https://arxiv.org/pdf/1801.09269.pdf
    """

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

    def exp(self, tangent_vec, base_point):
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

    def log(self, point, base_point):
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
        sqrt_bp, inv_sqrt_bp = powermh(base_point, [0.5, -0.5])
        pdt = powermh(Matrices.mul(sqrt_bp, point, sqrt_bp), 0.5)
        sqrt_product = Matrices.mul(sqrt_bp, pdt, inv_sqrt_bp)
        transp_sqrt_product = Matrices.transpose(sqrt_product)
        return sqrt_product + transp_sqrt_product - 2 * base_point

    def squared_dist(self, point_a, point_b):
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

        Notes
        -----
        Use of `abs` in the output prevents nan when calling
        `sqrt` in very small negative outputs (e.g. -1e-16).
        """
        tr_a = gs.trace(point_a)
        tr_b = gs.trace(point_b)

        point_a_sqrt = apply_func_to_eigvalsh(point_a, gs.sqrt)

        c_eigvals = gs.linalg.eigvalsh(
          Matrices.mul(Matrices.mul(point_a_sqrt, point_b), point_a_sqrt)
        )
        cross_term = gs.sum(gs.sqrt(c_eigvals), axis=-1)

        return gs.abs(tr_a + tr_b - 2 * cross_term)

    def parallel_transport(
        self,
        tangent_vec,
        base_point,
        direction=None,
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
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at `base_point` to transport.
        base_point : array-like, shape=[..., n, n]
            Initial point of the geodesic.
        direction : array-like, shape=[..., n, n]
            Tangent vector at `base_point`, initial velocity of the geodesic to
            transport along.
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
            end_point = self.exp(direction, base_point)

        horizontal_lift_a = gs.linalg.solve_sylvester(
            base_point, base_point, tangent_vec
        )

        square_root_bp, inverse_square_root_bp = powermh(base_point, [0.5, -0.5])
        end_point_lift = Matrices.mul(square_root_bp, end_point, square_root_bp)
        square_root_lift = powermh(end_point_lift, 0.5)

        horizontal_velocity = gs.matmul(inverse_square_root_bp, square_root_lift)
        partial_horizontal_velocity = Matrices.mul(horizontal_velocity, square_root_bp)
        partial_horizontal_velocity = partial_horizontal_velocity + Matrices.transpose(
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


class SPDEuclideanMetric(MatricesMetric):
    """Class for the Euclidean metric on the SPD manifold."""

    @staticmethod
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
        invsqrt_base_point = powermh(base_point, -0.5)

        reduced_vec = gs.matmul(invsqrt_base_point, tangent_vec)
        reduced_vec = gs.matmul(reduced_vec, invsqrt_base_point)
        eigvals = gs.linalg.eigvalsh(reduced_vec)
        min_eig = gs.amin(eigvals, axis=-1)
        max_eig = gs.amax(eigvals, axis=-1)

        inf_value = gs.where(max_eig <= 0.0, gs.array(-math.inf), -1.0 / max_eig)
        sup_value = gs.where(min_eig >= 0.0, gs.array(-math.inf), -1.0 / min_eig)

        return gs.stack((inf_value, sup_value), axis=-1)

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


class SPDLogEuclideanMetric(PullbackDiffeoMetric):
    """Class for the Log-Euclidean metric on the SPD manifold."""

    def __init__(self, space, image_space=None):
        if image_space is None:
            image_space = SymmetricMatrices(n=space.n)
        diffeo = SymMatrixLog()
        super().__init__(space, diffeo, image_space)


class SPDPowerMetric(PullbackDiffeoMetric):
    r"""Pullback metric induced by the power diffeomorphism.

    Given an equipped image space, the pullback metric is given by

    .. math::

        g_{\Sigma}^{(p)}(X, X)=
        \frac{1}{p^2} g_{\Sigma^p}\left(d_{\Sigma}
        \operatorname{pow}_p(X), d_{\Sigma} \operatorname{pow}_p(X)\right)

    The image space must be equipped with a `ScalarProductMetric`. The scale
    :math:`s` relates with power :math:`p` by

    .. math::

        s = 1 / power^2

    Check section 5.3 of [T2022]_ for more details.

    References
    ----------
    .. [T2022] Thanwerdas, Y. (2022) Riemannian and stratified
        geometries of covariance and correlation matrices. Theses.
        Université Côte d’Azur. Available at:
        https://hal.archives-ouvertes.fr/tel-03698752 (Accessed: 29 September 2022).
    """

    def __init__(self, space, image_space):
        if not isinstance(image_space.metric, ScalarProductMetric):
            raise ValueError(
                "`image-space` must be equipped with a `ScalarProductMetric`"
            )
        power = 1 / gs.sqrt(image_space.metric.scale)
        diffeo = MatrixPower(power)

        super().__init__(space, diffeo, image_space)


class LieCholeskyMetric(PullbackDiffeoMetric):
    """Pullback metric via a diffeomorphism.

    Diffeormorphism between SPD matrices and PLT matrices equipped with
    left invariant metric (see chapter 7 [TP2022]_).

    References
    ----------
    .. [T2022] Yann Thanwerdas. Riemannian and stratified
        geometries on covariance and correlation matrices. Differential
        Geometry [math.DG]. Université Côte d'Azur, 2022.
    """

    def __init__(self, space):
        image_space = PositiveLowerTriangularMatrices(space.n, equip=False)
        image_space.equip_with_metric(InvariantPositiveLowerTriangularMatricesMetric)

        diffeo = CholeskyMap()

        super().__init__(space=space, diffeo=diffeo, image_space=image_space)
