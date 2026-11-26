"""The Siegel manifold.

The Siegel manifold is a generalization of the Poincaré disk to complex matrices
of singular values strictly smaller than one.
It is defined as the set of complex matrices M such that:
I - M @ M.conj().T is a positive definite matrix.
Warning: another more restrictive definition of the Siegel disk also exists
which adds a symmetry condition on the matrices.
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
    New theoretical tools in the Siegel space for vectorial
    autoregressive data classification,
    Geometric Science of Information, 2021.
    https://franknielsen.github.io/IG/GSI2021-SiegelLogExpClassification.pdf
.. [JV2016] B. Jeuris and R. Vandebril. The Kähler mean of Block-Toeplitz
    matrices with Toeplitz structured blocks, 2016.
    https://epubs.siam.org/doi/pdf/10.1137/15M102112X
"""

import functools
import math

import geomstats.backend as gs
from geomstats import algebra_utils as utils
from geomstats._einsum import scalarmatmul
from geomstats.geometry.base import VectorSpaceOpenSet
from geomstats.geometry.complex_matrices import ComplexMatrices, ComplexMatrixOperations
from geomstats.geometry.hermitian_matrices import apply_func_to_eigvalsh
from geomstats.geometry.homogeneous_metric import RiemannianSymmetricMetric
from geomstats.geometry.isometry import PointedIsometry
from geomstats.geometry.manifold import _get_default_dtype
from geomstats.geometry.matrices import MatrixOperations
from geomstats.geometry.symmetric_matrices import SymmetricMatrices


def _tanh_card(eigvals):
    """Evaluate tanh(sqrt(x)) / sqrt(x), including its limit at zero."""
    values = utils.taylor_exp_even_func(
        gs.maximum(eigvals, 0.0), utils.inv_tanh_close_0
    )
    return gs.cast(1 / values, eigvals.dtype)


def _arctanh_card(eigvals):
    """Evaluate arctanh(sqrt(x)) / sqrt(x), including its limit at zero."""
    values = utils.taylor_exp_even_func(
        gs.maximum(eigvals, 0.0), utils.arctanh_card_close_0
    )
    return gs.cast(values, eigvals.dtype)


class SiegelTransvection(PointedIsometry):
    r"""Transvection of the Siegel space sending an anchor to zero.

    Denoting the anchor by :math:`A`, the transvection is the isometry

    .. math::

        \phi_A(Z) = (I - AA^*)^{-1/2} (Z - A) (I - A^* Z)^{-1}
        (I - A^* A)^{1/2}

    along the geodesic from :math:`A` to the null matrix. Its inverse is
    :math:`\phi_{-A}`.

    Parameters
    ----------
    anchor : array-like, shape=[..., n, n]
        Point on the Siegel space sent to zero.
    """

    def __init__(self, anchor):
        super().__init__(anchor, gs.zeros(anchor.shape[-2:], dtype=anchor.dtype))
        self._anchor_transconj = ComplexMatrixOperations.transconjugate(anchor)
        self._identity = gs.eye(anchor.shape[-1], dtype=anchor.dtype)

    @functools.cached_property
    def _left_eigh(self):
        """Eigendecomposition of :math:`I - AA^*`."""
        return gs.linalg.eigh(
            self._identity - gs.matmul(self.anchor, self._anchor_transconj)
        )

    @functools.cached_property
    def _right_eigh(self):
        """Eigendecomposition of :math:`I - A^* A`."""
        return gs.linalg.eigh(
            self._identity - gs.matmul(self._anchor_transconj, self.anchor)
        )

    @staticmethod
    def _power(eigh, power):
        """Compute a power of a Hermitian matrix from its eigendecomposition."""
        eigvals, eigvecs = eigh
        return gs.matmul(
            eigvecs * gs.power(eigvals, power)[..., None, :],
            ComplexMatrixOperations.transconjugate(eigvecs),
        )

    @functools.cached_property
    def _left_factor(self):
        """Compute :math:`(I - AA^*)^{-1/2}`."""
        return self._power(self._left_eigh, -1 / 2)

    @functools.cached_property
    def _inv_left_factor(self):
        """Compute :math:`(I - AA^*)^{1/2}`."""
        return self._power(self._left_eigh, 1 / 2)

    @functools.cached_property
    def _right_factor(self):
        """Compute :math:`(I - A^* A)^{1/2}`."""
        return self._power(self._right_eigh, 1 / 2)

    @functools.cached_property
    def _inv_right_factor(self):
        """Compute :math:`(I - A^* A)^{-1/2}`."""
        return self._power(self._right_eigh, -1 / 2)

    def _map(self, point, sign):
        """Apply the transvection of anchor `sign * anchor`."""
        anchor = sign * self.anchor
        anchor_transconj = sign * self._anchor_transconj
        inv_aux = gs.linalg.inv(self._identity - gs.matmul(anchor_transconj, point))
        return MatrixOperations.mul(
            self._left_factor, point - anchor, inv_aux, self._right_factor
        )

    def _tangent(self, tangent_vec, point, sign):
        r"""Apply the differential at `point` of the transvection of `sign * anchor`.

        With :math:`M = I - A^* Z`, the differential is

        .. math::

            d\phi_A(Z)[W] = (I - AA^*)^{-1/2} (I + (Z - A) M^{-1} A^*)
            W M^{-1} (I - A^* A)^{1/2}
        """
        anchor = sign * self.anchor
        anchor_transconj = sign * self._anchor_transconj
        inv_aux = gs.linalg.inv(self._identity - gs.matmul(anchor_transconj, point))
        aux = self._identity + MatrixOperations.mul(
            point - anchor, inv_aux, anchor_transconj
        )
        return MatrixOperations.mul(
            self._left_factor, aux, tangent_vec, inv_aux, self._right_factor
        )

    def __call__(self, base_point):
        """Send a point by the transvection.

        Parameters
        ----------
        base_point : array-like, shape=[..., n, n]
            Point on the Siegel space.

        Returns
        -------
        image_point : array-like, shape=[..., n, n]
            Image of the point.
        """
        return self._map(base_point, 1.0)

    def inverse(self, image_point):
        """Send a point by the inverse transvection.

        Parameters
        ----------
        image_point : array-like, shape=[..., n, n]
            Point on the Siegel space.

        Returns
        -------
        base_point : array-like, shape=[..., n, n]
            Preimage of the point.
        """
        return self._map(image_point, -1.0)

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        """Send a tangent vector by the differential of the transvection.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.
        image_point : array-like, shape=[..., n, n]
            Image of the base point. Used if `base_point` is not given.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at the image of the base point.
        """
        if base_point is None:
            if image_point is None:
                raise ValueError("Either a base point or an image point is needed.")
            base_point = self.inverse(image_point)
        return self._tangent(tangent_vec, base_point, 1.0)

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
        """Send a tangent vector by the differential of the inverse transvection.

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at image point.
        image_point : array-like, shape=[..., n, n]
            Image point.
        base_point : array-like, shape=[..., n, n]
            Preimage of the image point. Used if `image_point` is not given.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at the preimage of the image point.
        """
        if image_point is None:
            if base_point is None:
                raise ValueError("Either a base point or an image point is needed.")
            image_point = self(base_point)
        return self._tangent(image_tangent_vec, image_point, -1.0)

    def tangent_at_anchor(self, tangent_vec):
        r"""Send a tangent vector at the anchor by the differential.

        At the anchor, the differential of the transvection reduces to

        .. math::

            d\phi_A(A)[W] = (I - AA^*)^{-1/2} W (I - A^* A)^{-1/2}

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at the anchor.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at zero.
        """
        return MatrixOperations.mul(
            self._left_factor, tangent_vec, self._inv_right_factor
        )

    def inverse_tangent_at_reference(self, image_tangent_vec):
        r"""Send a tangent vector at zero by the differential of the inverse.

        At zero, the differential of the inverse transvection reduces to

        .. math::

            d\phi_{-A}(0)[V] = (I - AA^*)^{1/2} V (I - A^* A)^{1/2}

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at zero.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at the anchor.
        """
        return MatrixOperations.mul(
            self._inv_left_factor, image_tangent_vec, self._right_factor
        )


class Siegel(VectorSpaceOpenSet):
    """Class for the Siegel space.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    symmetric : bool
        If symmetric is True, add a symmetry condition
        on the matrices to belong to the Siegel space.
    """

    def __init__(self, n, symmetric=False, equip=True):
        embedding_space = (
            SymmetricMatrices(n, field="complex")
            if symmetric
            else ComplexMatrices(m=n, n=n)
        )
        super().__init__(
            dim=embedding_space.dim,
            embedding_space=embedding_space,
            equip=equip,
        )
        self.n = n
        self.symmetric = symmetric

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return SiegelMetric

    def belongs(self, point, atol=gs.atol):
        """Check if a matrix belongs to the Siegel space.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point to be checked.
        atol : float
            Tolerance.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if point belongs to the Siegel space.
        """
        point_transconj = ComplexMatrixOperations.transconjugate(point)
        aux = gs.matmul(point, point_transconj)

        belongs = gs.all(gs.linalg.eigvalsh(aux) <= 1 + atol, axis=-1)
        return gs.logical_and(belongs, self.embedding_space.belongs(point, atol))

    def projection(self, point, atol=gs.atol):
        """Project a matrix to the Siegel space.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix to project.
        atol : float
            Tolerance.

        Returns
        -------
        projected : array-like, shape=[..., n, n]
            Matrix in the Siegel space.
        """
        point = self.embedding_space.projection(point)

        point_transconj = ComplexMatrixOperations.transconjugate(point)
        aux = gs.matmul(point, point_transconj)

        eigenvalues = gs.linalg.eigvalsh(aux)
        max_eigenvalues = gs.amax(eigenvalues, axis=-1) ** 0.5

        scalars = gs.where(
            (max_eigenvalues > 1.0 - gs.atol), (1 - atol) / max_eigenvalues, 1.0
        )
        return scalarmatmul(scalars, point)

    def random_point(self, n_samples=1, bound=1.0):
        """Generate random points in the Siegel space.

        The Siegel space is the set of complex matrices of singular values
        strictly smaller than one.
        The Frobenius norm of a matrix is greater than or equal to its spectral
        norm, i.e. its largest singular value.
        Hence, a matrix with Frobenius norm strictly smaller than one has all its
        singular values strictly smaller than one, and belongs to the Siegel disk.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        bound : float
            Bound of the interval in which to sample.

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
        return self.embedding_space.projection(samples)

    def random_tangent_vec(self, base_point, n_samples=1):
        """Sample on the tangent space of Siegel space from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        base_point : array-like, shape=[..., n, n]
            Base point of the tangent space.

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

        return self.to_tangent(samples, base_point)


class SiegelMetric(RiemannianSymmetricMetric):
    """Class for the Siegel metric.

    The Siegel metric is the metric of a Riemannian symmetric space.

    The geometry is only written at the reference point, the null matrix,
    and is moved to any other base point by the transvection sending that
    base point to the reference point. The inner product and the squared
    distance have closed forms that do not build the transvections.

    Parameters
    ----------
    space : Siegel
        Manifold.
    """

    @property
    def reference_point(self):
        """Reference point, the null matrix.

        Returns
        -------
        reference_point : array-like, shape=[n, n]
            Null matrix.
        """
        return gs.zeros(
            self._space.shape, dtype=_get_default_dtype(self._space.complex_repr)
        )

    def isometry_to_reference(self, base_point):
        """Get the isometry sending a base point to the reference point.

        Parameters
        ----------
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        isometry : SiegelTransvection
            Transvection sending the base point to the null matrix.
        """
        return SiegelTransvection(base_point)

    def inner_product_at_reference(self, tangent_vec_a, tangent_vec_b):
        """Compute the inner product at the reference point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at the reference point.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at the reference point.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner product.
        """
        return gs.real(
            ComplexMatrixOperations.frobenius_product(tangent_vec_a, tangent_vec_b)
        )

    def exp_at_reference(self, tangent_vec):
        r"""Compute the exponential map at the reference point.

        Denoting by :math:`V` the tangent vector and :math:`S = (VV^*)^{1/2}`,

        .. math::

            \exp_0(V) = \tanh(S) S^{-1} V

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at the reference point.

        Returns
        -------
        exp : array-like, shape=[..., n, n]
            Point on the manifold.
        """
        tangent_vec_transconj = ComplexMatrixOperations.transconjugate(tangent_vec)
        squared_radius = gs.matmul(tangent_vec, tangent_vec_transconj)
        factor = apply_func_to_eigvalsh(squared_radius, _tanh_card)
        return gs.matmul(factor, tangent_vec)

    def log_at_reference(self, point):
        r"""Compute the logarithm map at the reference point.

        Denoting by :math:`P` the point and :math:`R = (PP^*)^{1/2}`,

        .. math::

            \log_0(P) = \operatorname{artanh}(R) R^{-1} P

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point on the manifold.

        Returns
        -------
        log : array-like, shape=[..., n, n]
            Tangent vector at the reference point.
        """
        point_transconj = ComplexMatrixOperations.transconjugate(point)
        squared_radius = gs.matmul(point, point_transconj)
        factor = apply_func_to_eigvalsh(squared_radius, _arctanh_card)
        return gs.matmul(factor, point)

    def squared_dist_at_reference(self, point):
        r"""Compute the squared distance from the reference point.

        Denoting by :math:`\sigma_i` the singular values of the point,

        .. math::

            d(0, Z)^2 = \sum_i \operatorname{artanh}(\sigma_i)^2

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point on the manifold.

        Returns
        -------
        sq_dist : array-like, shape=[...,]
            Squared distance between the null matrix and the point.
        """
        point_transconj = ComplexMatrixOperations.transconjugate(point)
        eigvals = gs.linalg.eigvalsh(gs.matmul(point, point_transconj))
        singular_values = gs.sqrt(gs.maximum(eigvals, 0.0))
        return gs.sum(gs.arctanh(singular_values) ** 2, axis=-1)

    def sectional_curvature_at_reference(
        self, tangent_vec_a, tangent_vec_b, atol=gs.atol
    ):
        r"""Compute the sectional curvature at the reference point.

        The tangent vectors are first orthonormalized. For orthonormal
        :math:`X, Y`, the sectional curvature is

        .. math::

            K(X, Y) = -\frac{1}{2} \left(\|XY^* - YX^*\|^2
            + \|X^*Y - Y^*X\|^2\right)

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at the reference point.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at the reference point.
        atol : float
            Tolerance under which a vector is considered to be zero.

        Returns
        -------
        sectional_curvature : array-like, shape=[...,]
            Sectional curvature at the reference point.
        """
        dtype = tangent_vec_a.dtype

        def _normalize(tangent_vec):
            norm = gs.sqrt(self.inner_product_at_reference(tangent_vec, tangent_vec))
            scalars = gs.where(norm < atol, 0.0, 1 / norm)
            return scalarmatmul(gs.cast(scalars, dtype), tangent_vec)

        tangent_vec_a = _normalize(tangent_vec_a)
        inner_prod = self.inner_product_at_reference(tangent_vec_a, tangent_vec_b)
        tangent_vec_b = _normalize(
            tangent_vec_b - scalarmatmul(gs.cast(inner_prod, dtype), tangent_vec_a)
        )

        tangent_vec_a_transconj = ComplexMatrixOperations.transconjugate(tangent_vec_a)
        tangent_vec_b_transconj = ComplexMatrixOperations.transconjugate(tangent_vec_b)
        term_1 = gs.matmul(tangent_vec_a, tangent_vec_b_transconj) - gs.matmul(
            tangent_vec_b, tangent_vec_a_transconj
        )
        term_2 = gs.matmul(tangent_vec_a_transconj, tangent_vec_b) - gs.matmul(
            tangent_vec_b_transconj, tangent_vec_a
        )
        return -0.5 * (
            gs.linalg.matrix_norm(term_1) ** 2 + gs.linalg.matrix_norm(term_2) ** 2
        )

    def injectivity_radius_at_reference(self):
        """Compute the injectivity radius at the reference point.

        The Siegel space is a Hadamard manifold, so the radius is infinite.

        Returns
        -------
        radius : array-like, shape=[]
            Injectivity radius.
        """
        return gs.array(math.inf)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        r"""Compute the inner product at a base point.

        Denoting the base point by :math:`B`,

        .. math::

            \langle V, W \rangle_B = \operatorname{Re} \operatorname{tr}
            \left((I - BB^*)^{-1} V (I - B^* B)^{-1} W^*\right)

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
            Inner product.
        """
        identity = gs.eye(base_point.shape[-1], dtype=base_point.dtype)
        base_point_transconj = ComplexMatrixOperations.transconjugate(base_point)
        aux_a = gs.matmul(
            gs.linalg.inv(identity - gs.matmul(base_point, base_point_transconj)),
            tangent_vec_a,
        )
        aux_b = gs.matmul(
            gs.linalg.inv(identity - gs.matmul(base_point_transconj, base_point)),
            ComplexMatrixOperations.transconjugate(tangent_vec_b),
        )
        return gs.real(MatrixOperations.trace_product(aux_a, aux_b))

    def squared_dist(self, point_a, point_b):
        r"""Compute the squared distance between two points.

        Denoting the points by :math:`A` and :math:`B`, the squared distance is
        :math:`\sum_i \operatorname{artanh}(\sigma_i)^2`, where the
        :math:`\sigma_i` are the singular values of the image of :math:`B` by
        the transvection sending :math:`A` to zero. They are the singular
        values of :math:`C^{-1} (B - A) (I - A^* B)^{-1} D`, where :math:`C`
        and :math:`D` are the Cholesky factors of :math:`I - AA^*` and
        :math:`I - A^* A`.

        Parameters
        ----------
        point_a : array-like, shape=[..., n, n]
            Point on the manifold.
        point_b : array-like, shape=[..., n, n]
            Point on the manifold.

        Returns
        -------
        sq_dist : array-like, shape=[...,]
            Squared distance.
        """
        identity = gs.eye(point_a.shape[-1], dtype=point_a.dtype)
        point_a_transconj = ComplexMatrixOperations.transconjugate(point_a)
        left_chol = gs.linalg.cholesky(
            identity - gs.matmul(point_a, point_a_transconj)
        )
        right_chol = gs.linalg.cholesky(
            identity - gs.matmul(point_a_transconj, point_a)
        )
        aux = gs.matmul(
            gs.linalg.inv(identity - gs.matmul(point_a_transconj, point_b)),
            right_chol,
        )
        aux = MatrixOperations.mul(gs.linalg.inv(left_chol), point_b - point_a, aux)
        eigvals = gs.linalg.eigvalsh(
            gs.matmul(aux, ComplexMatrixOperations.transconjugate(aux))
        )
        singular_values = gs.sqrt(gs.maximum(eigvals, 0.0))
        return gs.sum(gs.arctanh(singular_values) ** 2, axis=-1)
