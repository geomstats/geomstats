"""Module exposing the `Matrices` and `MatricesMetric` class."""

from functools import reduce

import geomstats.backend as gs
import geomstats.errors
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.base import VectorSpace
from geomstats.geometry.euclidean import EuclideanMetric


class Matrices(VectorSpace):
    """Class for the space of matrices (m, n).

    Parameters
    ----------
    m, n : int
        Integers representing the shapes of the matrices: m x n.
    """

    def __init__(self, m, n, **kwargs):
        if 'default_point_type' not in kwargs.keys():
            kwargs['default_point_type'] = 'matrix'
        super(Matrices, self).__init__(
            shape=(m, n), metric=MatricesMetric(m, n), **kwargs)
        geomstats.errors.check_integer(n, 'n')
        geomstats.errors.check_integer(m, 'm')
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
        ndim = point.ndim
        if ndim == 1:
            return False
        mat_dim_1, mat_dim_2 = point.shape[-2:]
        belongs = (mat_dim_1 == self.m) and (mat_dim_2 == self.n)
        return belongs if ndim == 2 else gs.tile(
            gs.array([belongs]), [point.shape[0]])

    @staticmethod
    def equal(mat_a, mat_b, atol=gs.atol):
        """Test if matrices a and b are close.

        Parameters
        ----------
        mat_a : array-like, shape=[..., dim1, dim2]
            Matrix.
        mat_b : array-like, shape=[..., dim2, dim3]
            Matrix.
        atol : float
            Tolerance.
            Optional, default: backend atol.

        Returns
        -------
        eq : array-like, shape=[...,]
            Boolean evaluating if the matrices are close.
        """
        return gs.all(gs.isclose(mat_a, mat_b, atol=atol), (-2, -1))

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

    @staticmethod
    def diagonal(mat):
        """Return the diagonal of a matrix as a vector.

        Parameters
        ----------
        mat : array-like, shape=[..., m, n]
            Matrix.

        Returns
        -------
        diagonal : array-like, shape=[..., min(m, n)]
            Vector of diagonal coefficients.
        """
        return gs.diagonal(mat, axis1=-2, axis2=-1)

    @staticmethod
    def is_square(mat):
        """Check if a matrix is square.

        Parameters
        ----------
        mat : array-like, shape=[..., m, n]
            Matrix.

        Returns
        -------
        is_square : array-like, shape=[...,]
            Boolean evaluating if the matrix is square.
        """
        n = mat.shape[-1]
        m = mat.shape[-2]
        return m == n

    @classmethod
    def is_symmetric(cls, mat, atol=gs.atol):
        """Check if a matrix is symmetric.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_sym : array-like, shape=[...,]
            Boolean evaluating if the matrix is symmetric.
        """
        is_square = cls.is_square(mat)
        if not is_square:
            is_vectorized = (gs.ndim(gs.array(mat)) == 3)
            return gs.array([False] * len(mat)) if is_vectorized else False
        return cls.equal(mat, cls.transpose(mat), atol)

    @classmethod
    def is_skew_symmetric(cls, mat, atol=gs.atol):
        """Check if a matrix is skew symmetric.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

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

    @classmethod
    def is_diagonal(cls, mat, atol=gs.atol):
        """Check if a matrix is square and diagonal.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_diagonal : array-like, shape=[...,]
            Boolean evaluating if the matrix is square and diagonal.
        """
        is_square = cls.is_square(mat)
        if not gs.all(is_square):
            return False
        diagonal_mat = from_vector_to_diagonal_matrix(cls.diagonal(mat))
        is_diagonal = gs.all(
            gs.isclose(mat, diagonal_mat, atol=atol), axis=(-2, -1))
        return is_diagonal

    def random_point(self, n_samples=1, bound=1.):
        """Sample from a uniform distribution in a cube.

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
        return gs.einsum(
            '...ij,...ij->...', mat_1, mat_2)

    @staticmethod
    def trace_product(mat_1, mat_2):
        """Compute trace of the product of two matrices.

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
        return gs.einsum(
            '...ij,...ji->...', mat_1, mat_2)


class MatricesMetric(EuclideanMetric):
    """Euclidean metric on matrices given by Frobenius inner-product.

    Parameters
    ----------
    m, n : int
        Integers representing the shapes of the matrices: m x n.
    """

    def __init__(self, m, n, **kwargs):
        dimension = m * n
        super(MatricesMetric, self).__init__(
            dim=dimension, default_point_type='matrix')

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
        return Matrices.frobenius_product(tangent_vec_a, tangent_vec_b)

    def norm(self, vector, base_point=None):
        """Compute norm of a matrix.

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
        return gs.linalg.norm(vector, axis=(-2, -1))
