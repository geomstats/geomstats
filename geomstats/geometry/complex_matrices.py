"""Module exposing the `ComplexMatrices` class.

Lead author: Yann Cabanes.
"""

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.base import ComplexVectorSpace
from geomstats.geometry.hermitian import HermitianMetric
from geomstats.geometry.matrices import Matrices
from geomstats.vectorization import repeat_out


class ComplexMatrices(ComplexVectorSpace):
    """Class for the space of complex matrices (m, n).

    Parameters
    ----------
    m, n : int
        Integers representing the shapes of the matrices: m x n.
    """

    def __init__(self, m, n, equip=True):
        geomstats.errors.check_integer(n, "n")
        geomstats.errors.check_integer(m, "m")
        super().__init__(shape=(m, n), equip=equip)
        self.m = m
        self.n = n

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return ComplexMatricesMetric

    def _create_basis(self):
        """Create the canonical basis."""
        cdtype = gs.get_default_cdtype()
        m, n = self.m, self.n
        return gs.reshape(gs.eye(n * m, dtype=cdtype), (n * m, m, n))

    def belongs(self, point, atol=gs.atol):
        """Check if point belongs to the Matrices space.

        Parameters
        ----------
        point : array-like, shape=[..., m, n]
            Point to be checked.
        atol : float
            Unused here.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the Matrices space.
        """
        is_matrix = super().belongs(point, atol=atol)
        belongs = gs.logical_and(is_matrix, gs.is_complex(point))
        return belongs

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
        return gs.transpose(gs.conj(mat), axes)

    @classmethod
    def is_hermitian(cls, mat, atol=gs.atol):
        """Check if a matrix is Hermitian.

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
        is_square = Matrices.is_square(mat)
        if not is_square:
            is_vectorized = gs.ndim(gs.array(mat)) == 3
            return gs.array([False] * len(mat)) if is_vectorized else False
        return Matrices.equal(mat, cls.transconjugate(mat), atol)

    @classmethod
    def is_hpd(cls, mat, atol=gs.atol):
        """Check if a matrix is Hermitian positive definite.

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
        is_hpd = gs.logical_and(cls.is_hermitian(mat, atol), Matrices.is_pd(mat))
        return is_hpd

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
        is_square = Matrices.is_square(mat)
        if not is_square:
            is_vectorized = gs.ndim(gs.array(mat)) == 3
            return gs.array([False] * len(mat)) if is_vectorized else False
        return Matrices.equal(mat, -cls.transconjugate(mat), atol)

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

        m, n = self.m, self.n
        size = (n_samples, m, n) if n_samples != 1 else (m, n)
        point = gs.cast(bound * (gs.random.rand(*size) - 0.5), dtype=cdtype)
        point += 1j * gs.cast(bound * (gs.random.rand(*size) - 0.5), dtype=cdtype)
        return point

    def random_tangent_vec(self, base_point, n_samples=1):
        """Generate random tangent vec.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        base_point :  array-like, shape=[..., dim]
            Point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vec at base point.
        """
        if (
            n_samples > 1
            and base_point.ndim > len(self.shape)
            and n_samples != len(base_point)
        ):
            raise ValueError(
                "The number of base points must be the same as the "
                "number of samples, when different from 1."
            )
        cdtype = gs.get_default_cdtype()

        tangent_vec = gs.cast(
            gs.random.normal(size=(n_samples,) + self.shape),
            dtype=cdtype,
        )
        tangent_vec += 1j * gs.cast(
            gs.random.normal(size=(n_samples,) + self.shape),
            dtype=cdtype,
        )
        return gs.squeeze(tangent_vec)

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
        return Matrices.mul(mat_2, mat_1, cls.transconjugate(mat_2))

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


class ComplexMatricesMetric(HermitianMetric):
    """Hermitian metric on complex matrices given by Frobenius inner-product."""

    @staticmethod
    def inner_product(tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute Frobenius inner-product of two tangent vectors.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., m, n]
            Tangent vector.
        tangent_vec_b : array-like, shape=[..., m, n]
            Tangent vector.
        base_point : array-like, shape=[..., m, n]
            Base point.
            Optional, default: None.

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Frobenius inner-product of tangent_vec_a and tangent_vec_b.
        """
        return ComplexMatrices.frobenius_product(tangent_vec_a, tangent_vec_b)

    def squared_norm(self, vector, base_point=None):
        """Compute the square of the norm of a vector.

        Squared norm of a vector associated to the inner product
        at the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        sq_norm : array-like, shape=[...,]
            Squared norm.
        """
        sq_norm = self.inner_product(vector, vector, base_point)
        sq_norm = gs.real(sq_norm)
        return sq_norm

    def norm(self, vector, base_point=None):
        """Compute norm of a complex matrix.

        Norm of a matrix associated to the Frobenius inner product.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        norm : array-like, shape=[...,]
            Norm.
        """
        out = gs.linalg.norm(vector, axis=(-2, -1))
        return repeat_out(self._space, out, vector, base_point)
