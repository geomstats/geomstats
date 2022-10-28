"""The Siegel manifold.

The Siegel manifold is a generalization of the Poincare disk to complex matrices
of singular values strictly lower than one.
It is defined as the set of complex matrices M such that:
I - M @ M.conj().T is a positive definite matrix.
Warning: another more restrictive definition of the Siegel disk also exists
which add a symmetry condition on the matrices.
It has been proven in [Cabanes2022]_ that the sub-manifold of symmetric Siegel
matrices is a totally geodesic sub-manifold of the Siegel space.
The sub-manifold of real Siegel matrices is also a totally geodesic sub-manifold
of the Siegel space.

Lead author: Yann Cabanes.

References
----------
.. [Cabanes2022] Yann Cabanes. Multidimensional complex stationary
    centered Gaussian autoregressive time series machine learning
    in Poincaré and Siegel disks: application for audio and radar
    clutter classification, PhD thesis, 2022
.. [Cabanes2021] Yann Cabanes and Frank Nielsen.
    New theoreticla tools in the Siegel space for vectorial
    autoregressive data classification,
    Geometric Science of Information, 2021.
    https://franknielsen.github.io/IG/GSI2021-SiegelLogExpClassification.pdf
.. [JV2016] B. Jeuris and R. Vandebril. The Kahler mean of Block-Toeplitz
    matrices with Toeplitz structured blocks, 2016.
    https://epubs.siam.org/doi/pdf/10.1137/15M102112X
"""

import geomstats.backend as gs
from geomstats.geometry.base import ComplexOpenSet
from geomstats.geometry.complex_matrices import ComplexMatrices
from geomstats.geometry.complex_riemannian_metric import ComplexRiemannianMetric
from geomstats.geometry.hermitian_matrices import HermitianMatrices


def _create_identity_mat(shape, dtype):
    """Stack identity matrices.

    Parameters
    ----------
    shape : tuple
        Desired identity matrix shape of form [..., n, n].
    dtype : dtype
        Desired dtype.

    Returns
    -------
    identity : array-like, shape=[..., n, n]
        Stacked identity matrices.
    """
    ndim = len(shape)
    if ndim == 2:
        return gs.eye(shape[-1], dtype=dtype)
    return gs.stack([gs.eye(shape[-1], dtype=dtype) for _ in range(shape[0])], axis=0)


class Siegel(ComplexOpenSet):
    """Class for the Siegel space.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    symmetric : bool
        If symmetric is True, add a symmetry condition
        on the matrices to belong to the Siegel space.
        Optional, default: False.
    scale : float
        Scale of the complex Poincare metric.
        Optional, default: 1.
    """

    def __init__(self, n, symmetric=False, scale=1.0, **kwargs):
        kwargs.setdefault("metric", SiegelMetric(n))
        super().__init__(
            dim=n**2, embedding_space=ComplexMatrices(m=n, n=n), **kwargs
        )
        self.n = n
        self.symmetric = symmetric
        self.scale = scale

    def belongs(self, point, atol=gs.atol):
        """Check if a matrix belongs to the Siegel space.

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
            Boolean denoting if mat belongs to the Siegel space.
        """
        point_transconj = ComplexMatrices.transconjugate(point)
        aux = gs.matmul(point, point_transconj)

        axis = -1 if gs.ndim(point) == 3 else None
        belongs = gs.all(gs.linalg.eigvalsh(aux) <= 1 - atol, axis=axis)

        if self.symmetric:
            return gs.logical_and(belongs, ComplexMatrices.is_symmetric(point))

        return belongs

    def projection(self, point, atol=gs.atol):
        """Project a matrix to the Siegel space.

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
            Matrix in the Siegel space.
        """
        if self.symmetric:
            point = ComplexMatrices.to_symmetric(point)

        point_transconj = ComplexMatrices.transconjugate(point)
        aux = gs.matmul(point, point_transconj)

        eigenvalues = gs.linalg.eigvalsh(aux)
        max_eigenvalues = gs.amax(eigenvalues, axis=-1) ** 0.5

        scalars = gs.where(
            (max_eigenvalues > 1.0 - gs.atol), (1 - atol) / max_eigenvalues, 1.0
        )
        return gs.einsum("...,...ij->...ij", scalars, point)

    def random_point(self, n_samples=1, bound=1.0):
        """Generate random points in the Siegel space.

        The Siegel space is the set of complex matrices of singular values
        strictly lower than one.
        The Frobenius norm of a matrix is greater than or equal to the spectral norm
        which corresponds to the largest singular value of a matrix.
        Then, simulating a matrix with Frobenius norm strictly lower than one,
        its singular values are also strictly lower than one,
        therefore this matrix belongs to the Siegel disk.

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
            Points sampled in the Siegel space.
        """
        n = self.n
        size = (n_samples, n, n) if n_samples != 1 else (n, n)

        samples = gs.random.rand(*size, dtype=gs.get_default_cdtype())
        samples -= 0.5 + 0.5j
        samples *= bound * (1 - gs.atol) * 2**0.5 / n
        return samples

    def random_tangent_vec(self, base_point=None, n_samples=1):
        """Sample on the tangent space of Siegel space from the uniform distribution.

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

        samples = gs.random.rand(*size, dtype=gs.get_default_cdtype())
        samples *= 2
        samples -= 1 + 1j

        return samples


class SiegelMetric(ComplexRiemannianMetric):
    """Class for the Siegel metric.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    scale : float
        Scale of the complex Poincare metric.
        Optional, default: 1.
    """

    def __init__(self, n, scale=1.0, **kwargs):
        dim = int(n**2)
        super().__init__(
            dim=dim,
            shape=(n, n),
            signature=(dim, 0),
        )
        self.n = n
        self.scale = scale

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the Siegel inner-product.

        Compute the inner-product of tangent_vec_a and tangent_vec_b
        at point base_point using the Siegel Riemannian metric.
        The expression of the inner product between the vectors`v` and `w`
        at point O is :math:`<v, w>_{O}
        = 1/2 * trace((I - O O^{H})^{-1} v (I - O^{H} O)^{-1} w^{H})
        + 1/2 * trace((I - O O^{H})^{-1} w (I - O^{H} O^{-1} v^{H})`
        where H denotes the conjugate transpose operator.

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
        identity = _create_identity_mat(base_point.shape, dtype=base_point.dtype)

        base_point_transconj = ComplexMatrices.transconjugate(base_point)
        tangent_vec_a_transconj = ComplexMatrices.transconjugate(tangent_vec_a)
        tangent_vec_b_transconj = ComplexMatrices.transconjugate(tangent_vec_b)

        aux_1 = gs.matmul(base_point, base_point_transconj)

        aux_2 = gs.matmul(base_point_transconj, base_point)

        aux_3 = identity - aux_1

        aux_4 = identity - aux_2

        inv_aux_3 = HermitianMatrices.powerm(aux_3, -1)

        inv_aux_4 = HermitianMatrices.powerm(aux_4, -1)

        aux_a = gs.matmul(inv_aux_3, tangent_vec_a)
        aux_b = gs.matmul(inv_aux_4, tangent_vec_b_transconj)
        trace_1 = ComplexMatrices.trace_product(aux_a, aux_b)

        aux_c = gs.matmul(inv_aux_3, tangent_vec_b)
        aux_d = gs.matmul(inv_aux_4, tangent_vec_a_transconj)
        trace_2 = ComplexMatrices.trace_product(aux_c, aux_d)

        inner_product = trace_1 + trace_2
        inner_product *= 0.5
        inner_product *= self.scale**2

        return inner_product

    @staticmethod
    def tangent_vec_from_base_point_to_zero(tangent_vec, base_point):
        """Transport a tangent vector from a base point to zero.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at zero.
        base_point : array-like, shape=[..., n, n]
            Point on the Siegel space.

        Returns
        -------
        tangent_vec_at_zero : array-like, shape=[..., n, n]
            Tangent vector at zero (null matrix).
        """
        identity = _create_identity_mat(base_point.shape, dtype=base_point.dtype)
        base_point_transconj = ComplexMatrices.transconjugate(base_point)
        aux_1 = gs.matmul(base_point, base_point_transconj)
        aux_2 = gs.matmul(base_point_transconj, base_point)
        aux_3 = identity - aux_1
        aux_4 = identity - aux_2
        factor_1 = HermitianMatrices.powerm(aux_3, -1 / 2)
        factor_3 = HermitianMatrices.powerm(aux_4, -1 / 2)
        prod_1 = gs.matmul(factor_1, tangent_vec)
        tangent_vec_at_zero = gs.matmul(prod_1, factor_3)
        return tangent_vec_at_zero

    @staticmethod
    def exp_at_zero(tangent_vec):
        """Compute the Siegel exponential map at zero.

        Compute the exponential map at zero (null matrix) of tangent_vec.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.

        Returns
        -------
        exp : array-like, shape=[..., n, n]
            Point on the manifold.
        """
        identity = _create_identity_mat(tangent_vec.shape, dtype=tangent_vec.dtype)
        tangent_vec_transconj = ComplexMatrices.transconjugate(tangent_vec)
        aux_1 = gs.matmul(tangent_vec, tangent_vec_transconj)
        aux_2 = HermitianMatrices.powerm(aux_1, 1 / 2)
        aux_3 = HermitianMatrices.expm(2 * aux_2)
        factor_1 = aux_3 - identity
        aux_4 = aux_3 + identity
        factor_2 = HermitianMatrices.powerm(aux_4, -1)
        factor_3 = HermitianMatrices.powerm(aux_2, -1)
        prod_1 = gs.matmul(factor_1, factor_2)
        prod_2 = gs.matmul(prod_1, factor_3)
        exp = gs.matmul(prod_2, tangent_vec)
        return exp

    @staticmethod
    def isometry(point, point_to_zero):
        """Define an isometry for the Siegel metric.

        Isometry for the Siegel metric sending point_to_zero
        (parameter of the isometry) on zero.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point on the Siegel space.
        point_to_zero : array-like, shape=[..., n, n]
            Point send on zero (null matrix) by the isometry.

        Returns
        -------
        point_image : array-like, shape=[..., n, n]
            Image of point by the isometry.
        """
        identity = _create_identity_mat(point.shape, dtype=point.dtype)
        point_to_zero_transconj = ComplexMatrices.transconjugate(point_to_zero)
        aux_1 = gs.matmul(point_to_zero, point_to_zero_transconj)
        aux_2 = gs.matmul(point_to_zero_transconj, point_to_zero)
        aux_3 = identity - aux_1
        aux_4 = identity - aux_2
        factor_1 = HermitianMatrices.powerm(aux_3, -1 / 2)
        factor_4 = HermitianMatrices.powerm(aux_4, 1 / 2)
        factor_2 = point - point_to_zero
        aux_5 = gs.matmul(point_to_zero_transconj, point)
        aux_6 = identity - aux_5
        factor_3 = gs.linalg.inv(aux_6)
        prod_1 = gs.matmul(factor_1, factor_2)
        prod_2 = gs.matmul(prod_1, factor_3)
        point_image = gs.matmul(prod_2, factor_4)
        return point_image

    def exp(self, tangent_vec, base_point):
        """Compute the Siegel exponential map.

        Compute the exponential map at base_point of tangent_vec.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Point on the manifold.

        Returns
        -------
        exp : array-like, shape=[..., n, n]
            Point on the manifold.
        """
        tangent_vec_at_zero = self.tangent_vec_from_base_point_to_zero(
            tangent_vec=tangent_vec, base_point=base_point
        )

        exp_zero = self.exp_at_zero(tangent_vec_at_zero)

        return self.isometry(point=exp_zero, point_to_zero=-base_point)

    @staticmethod
    def log_at_zero(point):
        """Compute the Siegel logarithm map at zero.

        Compute the logarithm map at zero (null matrix) of point.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point on the Siegel space.

        Returns
        -------
        log_at_zero : array-like, shape=[..., n, n]
            Riemannian logarithm at zero (null matrix).
        """
        identity = _create_identity_mat(point.shape, dtype=point.dtype)
        point_transconj = ComplexMatrices.transconjugate(point)
        aux_1 = gs.matmul(point, point_transconj)
        aux_2 = HermitianMatrices.powerm(aux_1, 1 / 2)
        num = identity + aux_2
        den = identity - aux_2
        inv_den = HermitianMatrices.powerm(den, -1)
        frac = gs.matmul(num, inv_den)
        factor_1 = gs.linalg.logm(frac)
        factor_2 = HermitianMatrices.powerm(aux_2, -1)
        prod_1 = gs.matmul(factor_1, factor_2)
        log_at_zero = gs.matmul(prod_1, point)
        log_at_zero *= 0.5
        return log_at_zero

    @staticmethod
    def tangent_vec_from_zero_to_base_point(tangent_vec, base_point):
        """Transport a tangent vector from zero to a base point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at zero.
        base_point : array-like, shape=[..., n, n]
            Point on the Siegel space.

        Returns
        -------
        tangent_vec_at_base_point : array-like, shape=[..., n, n]
            Tangent vector at the base point.
        """
        identity = _create_identity_mat(tangent_vec.shape, dtype=base_point.dtype)
        base_point_transconj = ComplexMatrices.transconjugate(base_point)
        aux_1 = gs.matmul(base_point, base_point_transconj)
        aux_2 = gs.matmul(base_point_transconj, base_point)
        aux_3 = identity - aux_1
        aux_4 = identity - aux_2
        factor_1 = HermitianMatrices.powerm(aux_3, 1 / 2)
        factor_3 = HermitianMatrices.powerm(aux_4, 1 / 2)
        prod_1 = gs.matmul(factor_1, tangent_vec)
        tangent_vec_at_base_point = gs.matmul(prod_1, factor_3)
        return tangent_vec_at_base_point

    def log(self, point, base_point):
        """Compute the Siegel logarithm map.

        Compute the Riemannian logarithm at point base_point
        of point wrt the metric defined in inner_product.
        Return a tangent vector at point base_point.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point on the Siegel space.
        base_point : array-like, shape=[..., n, n]
            Point on the Siegel space.

        Returns
        -------
        log : array-like, shape=[..., n, n]
            Riemannian logarithm at the base point.
        """
        point_at_zero = self.isometry(point=point, point_to_zero=base_point)

        logarithm_at_zero = self.log_at_zero(point_at_zero)

        return self.tangent_vec_from_zero_to_base_point(
            tangent_vec=logarithm_at_zero, base_point=base_point
        )

    def squared_dist(self, point_a, point_b):
        """Compute the Siegel squared distance.

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
        if gs.ndim(point_a) > gs.ndim(point_b):
            point_a, point_b = point_b, point_a

        identity = _create_identity_mat(point_b.shape, dtype=point_a.dtype)

        point_a_transconj = ComplexMatrices.transconjugate(point_a)
        point_b_transconj = ComplexMatrices.transconjugate(point_b)

        factor_1 = point_b - point_a

        factor_3 = ComplexMatrices.transconjugate(factor_1)

        aux_2 = gs.matmul(point_a_transconj, point_b)

        aux_3 = gs.matmul(point_a, point_b_transconj)

        aux_4 = identity - aux_2

        aux_5 = identity - aux_3

        factor_2 = gs.linalg.inv(aux_4)

        factor_4 = gs.linalg.inv(aux_5)

        prod = gs.einsum(
            "...ij,...jk,...kl,...lm->...im", factor_1, factor_2, factor_3, factor_4
        )

        point_b_shape = point_b.shape
        point_shape = point_b.shape[-2:]

        if len(point_b_shape) == 2:
            n_samples = 1
        else:
            n_samples = point_b_shape[0]

        prod = gs.reshape(prod, (n_samples, *point_shape))

        prod_power_one_half = []

        for i_sample in range(n_samples):
            if gs.all(prod[i_sample, ...] == 0):
                prod_power_one_half.append(gs.zeros(point_shape, dtype=point_a.dtype))
            else:
                prod_power_one_half.append(
                    gs.linalg.fractional_matrix_power(prod[i_sample], 1 / 2)
                )
        prod_power_one_half = gs.stack(prod_power_one_half)

        prod_power_one_half = gs.reshape(prod_power_one_half, point_b_shape)

        num = identity + prod_power_one_half
        den = identity - prod_power_one_half

        inv_den = gs.linalg.inv(den)

        frac = gs.matmul(num, inv_den)

        logarithm = gs.linalg.logm(frac)

        sq_dist = ComplexMatrices.trace_product(logarithm, logarithm)
        sq_dist *= 0.25
        sq_dist = gs.real(sq_dist)
        sq_dist *= self.scale**2
        sq_dist = gs.maximum(sq_dist, 0)
        return sq_dist
