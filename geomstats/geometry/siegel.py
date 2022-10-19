"""The Siegel space.

The Siegel disk representation is used.
The Siegel disk is a generalization of the Poincare disk in higher dimension.
Warning: another more restrictive definition of the Siegel disk also exists.
It is defined as the set of complex matrices M such that:
I - M @ M.conj().T is a positive definite matrix.
See [JV2016] for more details.

Lead author: Yann Cabanes.

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

# import autograd
# from scipy.linalg import fractional_matrix_power

import geomstats.backend as gs
from geomstats.geometry.base import OpenSet
from geomstats.geometry.complex_matrices import ComplexMatrices
from geomstats.geometry.complex_riemannian_metric import ComplexRiemannianMetric
from geomstats.geometry.hermitian_matrices import HermitianMatrices

CDTYPE = gs.get_default_cdtype()


class Siegel(OpenSet):
    """Class for the Siegel disk.

    The Siegel disk is a generalization of the complex Poincare disk
    to complex matrices with singular values lower than one.
    """

    def __init__(self, n, symmetric=False, scale=1, **kwargs):
        """Construct the Siegel disk.

        Parameters
        ----------
        n : int
            Integer representing the shape of the matrices: n x n.
        """
        kwargs.setdefault("metric", SiegelMetric(n))
        super().__init__(
            dim=n**2, embedding_space=ComplexMatrices(m=n, n=n), **kwargs
        )
        self.n = n
        self.symmetric = symmetric
        self.scale = scale

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the Siegel space.

        Evaluate if a point belongs to the Siegel space,
        i.e. evaluate if:
        I - M @ M.conj().T > 0.

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
            Boolean denoting if mat is belongs to the Siegel space.
        """
        point_shape = gs.shape(point)

        ndim = len(point_shape)
        if ndim == 3:
            n_samples = point_shape[0]

        point_transconj = ComplexMatrices.transconjugate(point)
        aux = gs.einsum("...ij,...jk->...ik", point, point_transconj)

        if ndim == 2:
            eigenvalues = gs.linalg.eigvalsh(aux)
            belongs = (eigenvalues <= 1 - atol).all()
        elif ndim == 3:
            belongs = gs.zeros([n_samples])
            for i_sample in range(n_samples):
                eigenvalues = gs.linalg.eigvalsh(aux[i_sample, ...])
                belongs[i_sample] = (eigenvalues < 1 - atol).all()

        if self.symmetric:
            belongs = gs.logical_and(belongs, ComplexMatrices.is_symmetric(point))

        return belongs

    def projection(self, point, atol=gs.atol):
        """Project a matrix to the Siegel space.
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
        data_type = point.dtype
        point_shape = gs.shape(point)

        ndim = len(point_shape)
        if ndim == 3:
            n_samples = point_shape[0]

        point_transconj = ComplexMatrices.transconjugate(point)
        aux = gs.einsum("...ij,...jk->...ik", point, point_transconj)

        projected = point
        eigenvalues = gs.linalg.eigvalsh(aux)
        max_eigenvalues = gs.max(eigenvalues, axis=-1) ** 0.5

        if ndim == 2:
            if max_eigenvalues >= 1 - atol:
                projected /= gs.cast((1 - atol) * max_eigenvalues, dtype=data_type)

        elif ndim == 3:
            for i_sample in range(n_samples):
                if max_eigenvalues[i_sample] > 1 - atol:
                    projected /= gs.cast(
                        (1 - atol) * max_eigenvalues[i_sample], dtype=data_type
                    )

        if self.symmetric:
            projected = ComplexMatrices.to_symmetric(projected)

        return projected

    def random_point(self, n_samples=1, bound=1.0):
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

        mat = gs.cast(gs.random.rand(*size), dtype=CDTYPE)
        mat += 1j * gs.cast(gs.random.rand(*size), dtype=CDTYPE)
        mat *= 2
        mat -= 1 + 1j
        mat *= bound
        siegel_mat = self.projection(mat)
        return siegel_mat

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

        tangent_vec = gs.cast(gs.random.rand(*size), dtype=CDTYPE)
        tangent_vec += 1j * gs.cast(gs.random.rand(*size), dtype=CDTYPE)
        tangent_vec *= 2
        tangent_vec -= 1 + 1j

        return tangent_vec


class SiegelMetric(ComplexRiemannianMetric):
    """Class for the Siegel metric."""

    def __init__(self, n, scale=1, **kwargs):
        """Construct the Siegel metric."""
        dim = int(n**2)
        super().__init__(
            dim=dim,
            shape=(n, n),
            signature=(dim, 0),
        )
        self.n = n
        assert scale > 0, "The scale should be strictly positive"
        self.scale = scale

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the Information Geometry inner-product.

        Compute the inner-product of tangent_vec_a and tangent_vec_b
        at point base_point using the Information Geometry Riemannian metric.

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

        data_type = (tangent_vec_a + tangent_vec_b + base_point).dtype

        identity = gs.zeros(base_point.shape, dtype=data_type)
        gs.einsum("...ii -> ...i", identity)[...] = 1

        base_point_transconj = ComplexMatrices.transconjugate(base_point)
        tangent_vec_a_transconj = ComplexMatrices.transconjugate(tangent_vec_a)
        tangent_vec_b_transconj = ComplexMatrices.transconjugate(tangent_vec_b)

        aux_1 = gs.einsum("...ij,...jk->...ik", base_point, base_point_transconj)

        aux_2 = gs.einsum("...ij,...jk->...ik", base_point_transconj, base_point)

        aux_3 = identity - aux_1

        aux_4 = identity - aux_2

        inv_aux_3 = HermitianMatrices.powerm(aux_3, -1)

        inv_aux_4 = HermitianMatrices.powerm(aux_4, -1)

        aux_a = gs.einsum("...ij,...jk->...ik", inv_aux_3, tangent_vec_a)
        aux_b = gs.einsum("...ij,...jk->...ik", inv_aux_4, tangent_vec_b_transconj)
        prod_1 = gs.einsum("...ij,...jk->...ik", aux_a, aux_b)
        trace_1 = 0.5 * gs.trace(prod_1)

        aux_c = gs.einsum("...ij,...jk->...ik", inv_aux_3, tangent_vec_b)
        aux_d = gs.einsum("...ij,...jk->...ik", inv_aux_4, tangent_vec_a_transconj)
        prod_2 = gs.einsum("...ij,...jk->...ik", aux_c, aux_d)
        trace_2 = 0.5 * gs.trace(prod_2)

        inner_product = trace_1 + trace_2
        inner_product *= self.scale**2

        return inner_product

    def basis_vector(self, index, point_type="matrix", data_type="complex"):
        """Create a basis vector.

        Parameters
        ----------
        index : int
            Index of the basis vector to create.
        point_type : string
            If point_type is set to 'matrix', of matrix is returned.
            Else, a vector is returned.
        data_type = string
            The data type of the basis vector to create (int, float, complex...).

        Returns
        -------
        basis_vect : array-like, shape=[n, n] or shape=[n ** 2]
            Basis vector.
        """

        dim = self.n**2
        basis_vect = gs.zeros([dim], dtype=data_type)
        basis_vect[index] = 1
        if point_type == "matrix":
            basis_vect = gs.reshape(basis_vect, (self.n, self.n))
        return basis_vect

    def inner_product_matrix(self, base_point=None, base_point_type="matrix"):
        """Compute the inner product matrix.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, defaults to zeros if None.

        base_point_type : string
            If base_point_type is set to 'matrix', the base point is a matrix.
            If base_point_type is set to 'vector', the base point is a vector.

        Returns
        -------
        inner_prod_mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """

        data_type = base_point.dtype
        dim = self.n**2

        if base_point is None:
            base_point = gs.zeros((1, dim))

        if base_point_type == "vector":
            base_point = gs.reshape(base_point, (-1, self.n, self.n))

        inner_prod_mat = gs.zeros([dim, dim], dtype=data_type)
        for i_index in range(dim):
            basis_vector_i = self.basis_vector(
                i_index, point_type="matrix", data_type=data_type
            )
            for j_index in range(dim):
                basis_vector_j = self.basis_vector(
                    j_index, point_type="matrix", data_type=data_type
                )
                inner_prod_mat[i_index, j_index] = self.inner_product(
                    basis_vector_i, basis_vector_j, base_point
                )
        inner_prod_mat = inner_prod_mat.real
        return inner_prod_mat

    @staticmethod
    def tangent_vec_from_base_point_to_zero(tangent_vec, base_point):

        data_type = (tangent_vec + base_point).dtype
        identity = gs.zeros(base_point.shape, dtype=data_type)
        gs.einsum("...ii -> ...i", identity)[...] = 1

        base_point_transconj = ComplexMatrices.transconjugate(base_point)

        aux_1 = gs.einsum("...ij,...jk->...ik", base_point, base_point_transconj)

        aux_2 = gs.einsum("...ij,...jk->...ik", base_point_transconj, base_point)

        aux_3 = identity - aux_1

        aux_4 = identity - aux_2

        factor_1 = HermitianMatrices.powerm(aux_3, -1 / 2)

        factor_3 = HermitianMatrices.powerm(aux_4, -1 / 2)

        prod_1 = gs.einsum("...ij,...jk->...ik", factor_1, tangent_vec)

        tangent_vec_at_zero = gs.einsum("...ij,...jk->...ik", prod_1, factor_3)

        return tangent_vec_at_zero

    @staticmethod
    def exp_at_zero(tangent_vec):
        data_type = tangent_vec.dtype
        identity = gs.zeros(tangent_vec.shape, dtype=data_type)
        gs.einsum("...ii -> ...i", identity)[...] = 1
        tangent_vec_transconj = ComplexMatrices.transconjugate(tangent_vec)
        aux_1 = gs.einsum("...ij,...jk->...ik", tangent_vec, tangent_vec_transconj)
        aux_2 = HermitianMatrices.powerm(aux_1, 1 / 2)
        aux_3 = HermitianMatrices.expm(2 * aux_2)
        factor_1 = aux_3 - identity
        aux_4 = aux_3 + identity
        factor_2 = HermitianMatrices.powerm(aux_4, -1)
        factor_3 = HermitianMatrices.powerm(aux_2, -1)
        prod_1 = gs.einsum("...ij,...jk->...ik", factor_1, factor_2)
        prod_2 = gs.einsum("...ij,...jk->...ik", prod_1, factor_3)
        exp = gs.einsum("...ij,...jk->...ik", prod_2, tangent_vec)
        return exp

    @staticmethod
    def isometry(point, point_to_zero):
        data_type = (point + point_to_zero).dtype
        identity = gs.zeros(point.shape, dtype=data_type)
        gs.einsum("...ii -> ...i", identity)[...] = 1
        point_to_zero_transconj = ComplexMatrices.transconjugate(point_to_zero)
        aux_1 = gs.einsum("...ij,...jk->...ik", point_to_zero, point_to_zero_transconj)
        aux_2 = gs.einsum("...ij,...jk->...ik", point_to_zero_transconj, point_to_zero)
        aux_3 = identity - aux_1
        aux_4 = identity - aux_2
        factor_1 = HermitianMatrices.powerm(aux_3, -1 / 2)
        factor_4 = HermitianMatrices.powerm(aux_4, 1 / 2)
        factor_2 = point - point_to_zero
        aux_5 = gs.einsum("...ij,...jk->...ik", point_to_zero_transconj, point)
        aux_6 = identity - aux_5
        factor_3 = gs.linalg.inv(aux_6)
        prod_1 = gs.einsum("...ij,...jk->...ik", factor_1, factor_2)
        prod_2 = gs.einsum("...ij,...jk->...ik", prod_1, factor_3)
        point_image = gs.einsum("...ij,...jk->...ik", prod_2, factor_4)
        return point_image

    def exp(self, tangent_vec, base_point):
        """Compute the exponential map.

        Compute the Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the metric defined in inner_product.

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

        tangent_vec_at_zero = self.tangent_vec_from_base_point_to_zero(
            tangent_vec=tangent_vec, base_point=base_point
        )

        exp_zero = self.exp_at_zero(tangent_vec_at_zero)

        exp = self.isometry(point=exp_zero, point_to_zero=-base_point)

        return exp

    @staticmethod
    def log_at_zero(point):
        data_type = point.dtype
        identity = gs.zeros(point.shape, dtype=data_type)
        gs.einsum("...ii -> ...i", identity)[...] = 1
        point_transconj = ComplexMatrices.transconjugate(point)
        aux_1 = gs.einsum("...ij,...jk->...ik", point, point_transconj)
        aux_2 = HermitianMatrices.powerm(aux_1, 1 / 2)
        num = identity + aux_2
        den = identity - aux_2
        inv_den = HermitianMatrices.powerm(den, -1)
        frac = gs.einsum("...ij,...jk->...ik", num, inv_den)
        factor_1 = gs.linalg.logm(frac)
        factor_2 = HermitianMatrices.powerm(aux_2, -1)
        prod_1 = gs.einsum("...ij,...jk->...ik", factor_1, factor_2)
        log = gs.einsum("...ij,...jk->...ik", prod_1, point)
        log *= 0.5
        return log

    @staticmethod
    def tangent_vec_from_zero_to_base_point(tangent_vec, base_point):
        data_type = (tangent_vec + base_point).dtype
        identity = gs.zeros(base_point.shape, dtype=data_type)
        gs.einsum("...ii -> ...i", identity)[...] = 1
        base_point_transconj = ComplexMatrices.transconjugate(base_point)
        aux_1 = gs.einsum("...ij,...jk->...ik", base_point, base_point_transconj)
        aux_2 = gs.einsum("...ij,...jk->...ik", base_point_transconj, base_point)
        aux_3 = identity - aux_1
        aux_4 = identity - aux_2
        factor_1 = HermitianMatrices.powerm(aux_3, 1 / 2)
        factor_3 = HermitianMatrices.powerm(aux_4, 1 / 2)
        prod_1 = gs.einsum("...ij,...jk->...ik", factor_1, tangent_vec)
        tangent_vec_at_base_point = gs.einsum("...ij,...jk->...ik", prod_1, factor_3)
        return tangent_vec_at_base_point

    def log(self, point, base_point):
        """Compute the Riemannian logarithm map.

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

        point_at_zero = self.isometry(point=point, point_to_zero=base_point)

        logarithm_at_zero = self.log_at_zero(point_at_zero)

        log = self.tangent_vec_from_zero_to_base_point(
            tangent_vec=logarithm_at_zero, base_point=base_point
        )

        return log

    def squared_norm(self, vector, base_point=None):
        """Compute the squared norm of a vector at a given base point.

        Squared norm of a vector associated with the inner product
        at the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]

        base_point : array-like, shape=[..., n, n]

        Returns
        -------
        sq_norm : array-like, shape=[..., n, n]
        """
        sq_norm = self.inner_product(vector, vector, base_point).real
        sq_norm = gs.maximum(sq_norm, 0)
        return sq_norm

    def squared_dist(self, point_a, point_b):
        """Compute the squared distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[..., n, n]

        point_b : array-like, shape=[..., n, n]

        Returns
        -------
        sq_dist : array-like, shape=[..., 1]

        References
        ----------
        The Kahler mean of Block-Toeplitz matrices
        with Toeplitz structured blocks
        B. Jeuris and R. Vandebril
        2016
        https://epubs.siam.org/doi/pdf/10.1137/15M102112X
        3.2. A second generalized transformation.
        """

        if len(point_a.shape) > len(point_b.shape):
            point_temp = point_a
            point_a = point_b
            point_b = point_temp

        data_type = (point_a + point_b).dtype
        identity = gs.zeros(point_b.shape, dtype=data_type)
        gs.einsum("...ii -> ...i", identity)[...] = 1

        point_a_transconj = ComplexMatrices.transconjugate(point_a)
        point_b_transconj = ComplexMatrices.transconjugate(point_b)

        factor_1 = point_b - point_a

        factor_3 = ComplexMatrices.transconjugate(factor_1)

        aux_2 = gs.einsum("...ij,...jk->...ik", point_a_transconj, point_b)

        aux_3 = gs.einsum("...ij,...jk->...ik", point_a, point_b_transconj)

        aux_4 = identity - aux_2

        aux_5 = identity - aux_3

        factor_2 = gs.linalg.inv(aux_4)

        factor_4 = gs.linalg.inv(aux_5)

        prod = gs.einsum(
            "...ij,...jk,...kl,...lm->...im", factor_1, factor_2, factor_3, factor_4
        )

        point_b_shape = point_b.shape
        point_shape = (point_b_shape[-2], point_b_shape[-1])

        if len(point_b_shape) == 2:
            n_samples = 1
        else:
            n_samples = point_b_shape[0]

        prod = gs.reshape(prod, (n_samples, point_b_shape[-2], point_b_shape[-1]))

        prod_power_one_half = gs.zeros(
            (n_samples, point_b_shape[-2], point_b_shape[-1]), dtype=data_type
        )

        for i_sample in range(n_samples):
            if gs.all(prod[i_sample, ...] == 0):
                prod_power_one_half[i_sample, ...] = gs.zeros(
                    point_shape, dtype=data_type
                )
            else:
                prod_power_one_half[i_sample, ...] = gs.linalg.fractional_matrix_power(
                    prod[i_sample], 1 / 2
                )

        prod_power_one_half = prod_power_one_half.reshape(point_b_shape)

        num = identity + prod_power_one_half
        den = identity - prod_power_one_half

        inv_den = gs.linalg.inv(den)

        frac = gs.einsum("...ij,...jk->...ik", num, inv_den)

        logarithm = gs.linalg.logm(frac)

        sq_logarithm = gs.einsum("...ij,...jk->...ik", logarithm, logarithm)
        sq_dist = gs.trace(sq_logarithm)

        sq_dist *= 0.25

        sq_dist = sq_dist.real
        sq_dist *= self.scale**2
        sq_dist = gs.maximum(sq_dist, 0)
        return sq_dist

    def dist(self, point_a, point_b):
        """Compute the geodesic distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[..., n, n]

        point_b : array-like, shape=[..., n, n]

        Returns
        -------
        dist : array-like, shape=[..., 1]
        """
        sq_dist = self.squared_dist(point_a, point_b)
        dist = sq_dist ** (1 / 2)
        return dist
