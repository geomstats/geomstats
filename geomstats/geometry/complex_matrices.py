"""Module exposing the `ComplexMatrices` class.

Lead author: Yann Cabanes.
"""

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.matrices import Matrices, MatricesMetric


class ComplexMatrices(Matrices):
    """Class for the space of complex matrices (m, n).

    Parameters
    ----------
    m, n : int
        Integers representing the shapes of the matrices: m x n.
    """

    def __init__(self, m, n, **kwargs):
        geomstats.errors.check_integer(n, "n")
        geomstats.errors.check_integer(m, "m")
        kwargs.setdefault("metric", MatricesMetric(m, n))
        kwargs.setdefault("default_point_type", "matrix")
        super(Matrices, self).__init__(shape=(m, n), **kwargs)
        self.m = m
        self.n = n

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
        is_matrix = super(ComplexMatrices, self).belongs(point, atol=gs.atol)
        is_complex = point.dtype in [gs.complex64, gs.complex128]
        if point.ndim >= 3 and point.shape[0] >= 2:
            is_matrix = list(is_matrix)
            belongs = is_matrix and is_complex
            belongs = gs.array(belongs)
        else:
            belongs = is_matrix and is_complex
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
        is_vectorized = gs.ndim(gs.array(mat)) == 3
        axes = (0, 2, 1) if is_vectorized else (1, 0)
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
        is_square = cls.is_square(mat)
        if not is_square:
            is_vectorized = gs.ndim(gs.array(mat)) == 3
            return gs.array([False] * len(mat)) if is_vectorized else False
        return cls.equal(mat, cls.transconjugate(mat), atol)

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
        print("\n")
        print("\n")
        print("\n")
        print(cls.is_hermitian(mat, atol))
        print("\n")
        print("\n")
        print("\n")
        print(cls.is_pd(mat))
        print("\n")
        print("\n")
        print("\n")
        is_hpd = gs.logical_and(cls.is_hermitian(mat, atol), cls.is_pd(mat))
        print("\n")
        print("\n")
        print("\n")
        print(is_hpd)
        return is_hpd

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
        m, n = self.m, self.n
        size = (n_samples, m, n) if n_samples != 1 else (m, n)
        point = gs.cast(bound * (gs.random.rand(*size) - 0.5), dtype=gs.complex128)
        point += 1j * gs.cast(
            bound * (gs.random.rand(*size) - 0.5), dtype=gs.complex128
        )
        point /= 2**0.5
        return point

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


class ComplexMatricesMetric(MatricesMetric):
    """Euclidean metric on complex matrices given by Frobenius inner-product.

    Parameters
    ----------
    m, n : int
        Integers representing the shapes of the matrices: m x n.
    """

    def __init__(self, m, n, **kwargs):
        super(ComplexMatricesMetric, self).__init__(m, n, **kwargs)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
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
