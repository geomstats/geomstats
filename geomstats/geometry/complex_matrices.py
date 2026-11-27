"""Module exposing the `ComplexMatrices` class.

Lead author: Yann Cabanes.
"""

import math

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.base import MatrixVectorSpace
from geomstats.geometry.connection import FlatMixin
from geomstats.geometry.matrices import MatrixOperations
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.vectorization import repeat_out


class ComplexMatrixOperations:
    """Operations on complex matrices."""

    @staticmethod
    def transconjugate(mat):
        """Return the transconjugate of matrices.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        transconjugate : array-like, shape=[..., n, n]
            Transconjugated matrix.
        """
        ndim = gs.ndim(mat)
        axes = list(range(0, ndim))
        axes[-1] = ndim - 2
        axes[-2] = ndim - 1
        return gs.matrix_transpose(gs.conj(mat))

    @classmethod
    def is_hermitian(cls, mat, atol=gs.atol):
        """Check if a square matrix is Hermitian.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_herm : array-like, shape=[...,]
            Boolean evaluating if the matrix is symmetric.
        """
        return MatrixOperations.equal(mat, cls.transconjugate(mat), atol)

    @classmethod
    def is_hpd(cls, mat, atol=gs.atol):
        """Check if a square matrix is Hermitian positive definite.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_hpd : array-like, shape=[...,]
            Boolean evaluating if the matrix is Hermitian positive definite.
        """
        return gs.logical_and(cls.is_hermitian(mat, atol), MatrixOperations.is_pd(mat))

    @classmethod
    def is_skew_hermitian(cls, mat, atol=gs.atol):
        """Check if a matrix is skew-Hermitian.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_skew_herm : array-like, shape=[...,]
            Boolean evaluating if the matrix is skew-Hermitian.
        """
        return MatrixOperations.equal(mat, -cls.transconjugate(mat), atol)

    @classmethod
    def to_hermitian(cls, mat):
        """Make a matrix Hermitian.

        Make a matrix Hermitian by averaging it
        with its transconjugate.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        herm : array-like, shape=[..., n, n]
            Hermitian matrix.
        """
        return 1 / 2 * (mat + cls.transconjugate(mat))

    @classmethod
    def to_skew_hermitian(cls, mat):
        """Make a matrix skew-Hermitian.

        Make matrix skew-Hermitian by averaging it
        with minus its transconjugate.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        skew_sym : array-like, shape=[..., n, n]
            Skew-Hermitian matrix.
        """
        return 1 / 2 * (mat - cls.transconjugate(mat))

    @classmethod
    def congruent(cls, mat_1, mat_2):
        r"""Compute the congruent action of mat_2 on mat_1.

        This is :math:`mat\_2 \ mat\_1 \ mat\_2^T`.

        Parameters
        ----------
        mat_1 : array-like, shape=[..., n, n]
            Matrix.
        mat_2 : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        cong : array-like, shape=[..., n, n]
            Result of the congruent action.
        """
        return MatrixOperations.mul(mat_2, mat_1, cls.transconjugate(mat_2))

    @staticmethod
    def frobenius_product(mat_1, mat_2):
        """Compute Frobenius inner-product of two matrices.

        The `einsum` function is used to avoid computing a matrix product. It
        is also faster than using a sum an element-wise product.

        Parameters
        ----------
        mat_1 : array-like, shape=[..., m, n]
            Matrix.
        mat_2 : array-like, shape=[..., m, n]
            Matrix.

        Returns
        -------
        product : array-like, shape=[...,]
            Frobenius inner-product of mat_1 and mat_2
        """
        return gs.einsum("...ij,...ij->...", gs.conj(mat_1), mat_2)


class ComplexMatrices(MatrixVectorSpace):
    """Class for the space of complex matrices (m, n).

    Parameters
    ----------
    m, n : int
        Integers representing the shapes of the matrices: m x n.
    equip : bool
        Whether to equip space with default metric.
    """

    def __init__(self, m, n, equip=True):
        geomstats.errors.check_integer(n, "n")
        geomstats.errors.check_integer(m, "m")
        super().__init__(shape=(m, n), equip=equip, field="complex")
        self.m = m
        self.n = n

    def _create_basis(self):
        """Create the canonical basis."""
        m, n = self.m, self.n
        return gs.reshape(gs.eye(n * m, dtype=gs.get_default_cdtype()), (n * m, m, n))

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return ComplexMatricesMetric

    def basis_representation(self, matrix_representation):
        """Compute the coefficients of matrices in the given basis.

        Parameters
        ----------
        matrix_representation : array-like, shape=[..., m, n]
            Matrix.

        Returns
        -------
        basis_representation : array-like, shape=[..., m * n]
            Coefficients in the basis.
        """
        return MatrixOperations.flatten(matrix_representation)

    def matrix_representation(self, basis_representation):
        """Compute the matrix representation for the given basis coefficients.

        The coefficients in the canonical basis are the flattened entries of
        the matrix, so this reshapes them into a matrix.

        Parameters
        ----------
        basis_representation : array-like, shape=[..., m * n]
            Coefficients in the basis.

        Returns
        -------
        matrix_representation : array-like, shape=[..., m, n]
            Matrix.
        """
        return gs.reshape(
            basis_representation, basis_representation.shape[:-1] + (self.m, self.n)
        )

    def random_point(self, n_samples=1, bound=1.0):
        """Sample from a uniform distribution in a complex cube.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample each entry.
            Optional, default: 1.

        Returns
        -------
        point : array-like, shape=[..., m, n]
            Sample.
        """
        cdtype = gs.get_default_cdtype()

        size = (n_samples,) + self.shape if n_samples != 1 else self.shape
        point = gs.cast(bound * (gs.random.rand(*size) - 0.5), dtype=cdtype)
        point += 1j * gs.cast(bound * (gs.random.rand(*size) - 0.5), dtype=cdtype)
        return point


class ComplexMatricesMetric(FlatMixin, RiemannianMetric):
    """Metric on complex matrices given by the Frobenius inner-product.

    The inner product is the real part of the Hermitian Frobenius product,
    and it is independent of the base point. The metric is flat, and
    geodesics are straight lines.

    Parameters
    ----------
    space : ComplexMatrices
        Manifold.
    """

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute Frobenius inner-product of two tangent vectors.

        It is the real part of the Hermitian Frobenius product.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., m, n]
            Tangent vector.
        tangent_vec_b : array-like, shape=[..., m, n]
            Tangent vector.
        base_point : array-like, shape=[..., m, n]
            Base point.

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Frobenius inner-product of tangent_vec_a and tangent_vec_b.
        """
        inner_prod = gs.real(
            ComplexMatrixOperations.frobenius_product(tangent_vec_a, tangent_vec_b)
        )
        return repeat_out(
            self._space.point_ndim, inner_prod, tangent_vec_a, tangent_vec_b, base_point
        )

    def squared_norm(self, vector, base_point=None):
        """Compute the square of the norm of a vector.

        Parameters
        ----------
        vector : array-like, shape=[..., m, n]
            Vector.
        base_point : array-like, shape=[..., m, n]
            Base point.

        Returns
        -------
        sq_norm : array-like, shape=[...,]
            Squared norm.
        """
        sq_norm = gs.linalg.matrix_norm(vector) ** 2
        return repeat_out(self._space.point_ndim, sq_norm, vector, base_point)

    def norm(self, vector, base_point=None):
        """Compute the norm of a vector.

        Parameters
        ----------
        vector : array-like, shape=[..., m, n]
            Vector.
        base_point : array-like, shape=[..., m, n]
            Base point.

        Returns
        -------
        norm : array-like, shape=[...,]
            Norm.
        """
        norm = gs.linalg.matrix_norm(vector)
        return repeat_out(self._space.point_ndim, norm, vector, base_point)

    def exp(self, tangent_vec, base_point):
        """Compute exp map of a base point in tangent vector direction.

        The Riemannian exponential is the addition.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., m, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., m, n]
            Base point.

        Returns
        -------
        exp : array-like, shape=[..., m, n]
            Riemannian exponential.
        """
        return base_point + tangent_vec

    def log(self, point, base_point):
        """Compute log map using a base point and other point.

        The Riemannian logarithm is the subtraction.

        Parameters
        ----------
        point : array-like, shape=[..., m, n]
            Point.
        base_point : array-like, shape=[..., m, n]
            Base point.

        Returns
        -------
        log : array-like, shape=[..., m, n]
            Riemannian logarithm.
        """
        return point - base_point

    def parallel_transport(
        self, tangent_vec, base_point=None, direction=None, end_point=None
    ):
        """Compute the parallel transport of a tangent vector.

        The parallel transport of a tangent vector is the vector itself.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., m, n]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., m, n]
            Point on the manifold. Point to transport from.
        direction : array-like, shape=[..., m, n]
            Tangent vector at base point, along which the parallel transport
            is computed.
        end_point : array-like, shape=[..., m, n]
            Point on the manifold. Point to transport to.

        Returns
        -------
        transported_tangent_vec : array-like, shape=[..., m, n]
            Transported tangent vector.
        """
        transported_tangent_vec = gs.copy(tangent_vec)
        return repeat_out(
            self._space.point_ndim,
            transported_tangent_vec,
            tangent_vec,
            base_point,
            direction,
            end_point,
            out_shape=self._space.shape,
        )

    def injectivity_radius(self, base_point=None):
        """Compute the radius of the injectivity domain.

        It is infinite everywhere.

        Parameters
        ----------
        base_point : array-like, shape=[..., m, n]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        radius = gs.array(math.inf)
        return repeat_out(self._space.point_ndim, radius, base_point)
