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
    def is_spd(cls, mat, atol=gs.atol):
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
        is_hpd = gs.logical_and(cls.is_hermitian(mat, atol), cls.is_pd(mat))
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
        point = bound * (gs.random.rand(*size) - 0.5)
        point += 1j * bound * (gs.random.rand(*size) - 0.5)
        point /= 2**0.5
        return point
