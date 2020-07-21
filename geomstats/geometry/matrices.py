"""Module exposing the `Matrices` and `MatricesMetric` class."""

from functools import reduce

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.manifold import Manifold


TOLERANCE = 1e-5


class Matrices(Manifold):
    """Class for the space of matrices (m, n).

    Parameters
    ----------
    m, n : int
        Integers representing the shapes of the matrices: m x n.
    """

    def __init__(self, m, n):
        super(Matrices, self).__init__(dim=m * n)
        geomstats.errors.check_integer(n, 'n')
        geomstats.errors.check_integer(m, 'm')
        self.m = m
        self.n = n
        self.default_point_type = 'matrix'
        self.metric = MatricesMetric(m, n)

    def belongs(self, point):
        """Check if point belongs to the Matrices space.

        Parameters
        ----------
        point : array-like, shape=[..., m, n]
            Point to be checked.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the Matrices space.
        """
        point = gs.to_ndarray(point, to_ndim=3)
        _, mat_dim_1, mat_dim_2 = point.shape
        return mat_dim_1 == self.m & mat_dim_2 == self.n

    @staticmethod
    def equal(mat_a, mat_b, atol=TOLERANCE):
        """Test if matrices a and b are close.

        Parameters
        ----------
        mat_a : array-like, shape=[..., dim1, dim2]
            Matrix.
        mat_b : array-like, shape=[..., dim2, dim3]
            Matrix.
        atol : float
            Tolerance.
            Optional, default: 1e-5.

        Returns
        -------
        eq : array-like, shape=[...,]
            Boolean evaluating if the matrices are close.
        """
        is_vectorized = \
            (gs.ndim(gs.array(mat_a)) == 3) or (gs.ndim(gs.array(mat_b)) == 3)
        axes = (1, 2) if is_vectorized else (0, 1)
        return gs.all(gs.isclose(mat_a, mat_b, atol=atol), axes)

    @staticmethod
    def mul(*args):
        """Compute the product of matrices a1, ..., an.

        Parameters
        ----------
        a1 : array-like, shape=[..., dim_1, dim_2]
            Matrix.
        a2 : array-like, shape=[..., dim_2, dim_3]
            Matrix.
        ...
        an : array-like, shape=[..., dim_n-1, dim_n]
            Matrix.

        Returns
        -------
        mul : array-like, shape=[..., dim_1, dim_n]
            Result of the product of matrices.
        """
        return reduce(gs.matmul, args)

    @classmethod
    def bracket(cls, mat_a, mat_b):
        """Compute the commutator of a and b, i.e. `[a, b] = ab - ba`.

        Parameters
        ----------
        mat_a : array-like, shape=[..., n, n]
            Matrix.
        mat_b : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        mat_c : array-like, shape=[..., n, n]
            Commutator.
        """
        return cls.mul(mat_a, mat_b) - cls.mul(mat_b, mat_a)

    @staticmethod
    def transpose(mat):
        """Return the transpose of matrices.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        transpose : array-like, shape=[..., n, n]
            Transposed matrix.
        """
        is_vectorized = (gs.ndim(gs.array(mat)) == 3)
        axes = (0, 2, 1) if is_vectorized else (1, 0)
        return gs.transpose(mat, axes)

    @classmethod
    def is_symmetric(cls, mat, atol=TOLERANCE):
        """Check if a matrix is symmetric.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.
        atol : float
            Absolute tolerance.
            Optional, default: 1e-5.

        Returns
        -------
        is_sym : array-like, shape=[...,]
            Boolean evaluating if the matrix is symmetric.
        """
        return cls.equal(mat, cls.transpose(mat), atol)

    @classmethod
    def is_skew_symmetric(cls, mat, atol=TOLERANCE):
        """Check if a matrix is skew symmetric.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.
        atol : float
            Absolute tolerance.
            Optional, default: 1e-5.

        Returns
        -------
        is_skew_sym : array-like, shape=[...,]
            Boolean evaluating if the matrix is skew-symmetric.
        """
        return cls.equal(mat, - cls.transpose(mat), atol)

    @classmethod
    def to_symmetric(cls, mat):
        """Make a matrix symmetric, by averaging with its transpose.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        sym : array-like, shape=[..., n, n]
            Symmetric matrix.
        """
        return 1 / 2 * (mat + cls.transpose(mat))

    @classmethod
    def to_skew_symmetric(cls, mat):
        """
        Make a matrix skew-symmetric, by averaging with minus its transpose.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        skew_sym : array-like, shape=[..., n, n]
            Skew-symmetric matrix.
        """
        return 1 / 2 * (mat - cls.transpose(mat))

    def random_uniform(self, n_samples=1, bound=1.):
        """Sample from a uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound.
            Optional, default: 1.

        Returns
        -------
        point : array-like, shape=[m, n] or [n_samples, m, n]
            Sample.
        """
        m, n = self.m, self.n
        size = (n_samples, m, n) if n_samples != 1 else (m, n)
        point = bound * (gs.random.rand(*size) - 0.5)
        return point

    @classmethod
    def congruent(cls, mat_1, mat_2):
        """Compute the congruent action of mat_2 on mat_1.

        This is :math: `mat_2 mat_1 mat_2^T`.

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
        return cls.mul(mat_2, mat_1, cls.transpose(mat_2))


class MatricesMetric(EuclideanMetric):
    """Euclidean metric on matrices given by Frobenius inner-product.

    Parameters
    ----------
    m, n : int
        Integers representing the shapes of the matrices: m x n.
    """

    def __init__(self, m, n):
        dimension = m * n
        super(MatricesMetric, self).__init__(
            dim=dimension)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute Frobenius inner-product of two tan vecs at `base_point`.

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
        inner_prod = gs.einsum(
            '...ij,...ij->...', tangent_vec_a, tangent_vec_b)
        return inner_prod
