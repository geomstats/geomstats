"""Module exposing the `Matrices` and `MatricesMetric` class."""

from functools import reduce

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric


TOLERANCE = 1e-5


class Matrices(Euclidean):
    """Class for the space of matrices (m, n)."""

    def __init__(self, m, n):
        assert isinstance(m, int) and isinstance(n, int) and m > 0 and n > 0
        super(Matrices, self).__init__(dimension=m * n)
        self.m = m
        self.n = n
        self.default_point_type = 'matrix'
        self.metric = MatricesMetric(m, n)

    def belongs(self, point):
        """Check if point belongs to the Matrix space."""
        point = gs.to_ndarray(point, to_ndim=3)
        _, mat_dim_1, mat_dim_2 = point.shape
        return mat_dim_1 == self.m & mat_dim_2 == self.n

    @staticmethod
    def equal(mat_a, mat_b, atol=TOLERANCE):
        """
        Test if matrices a and b are close.

        Parameters
        ----------
        mat_a : array-like, shape=[n_samples, dim1, dim2]
        mat_b : array-like, shape=[n_samples, dim2, dim3]

        Returns
        -------
        eq : array-like boolean, shape=[n_samples]
        """
        is_vectorized = \
            (gs.ndim(gs.array(mat_a)) == 3) or (gs.ndim(gs.array(mat_b)) == 3)
        axes = (1, 2) if is_vectorized else (0, 1)
        return gs.all(gs.isclose(mat_a, mat_b, atol=atol), axes)

    @staticmethod
    def mul(*args):
        """
        Return the product of matrices a1, ..., an.

        Parameters
        ----------
        a1 : array-like, shape=[n_samples, dim_1, dim_2]
        a2 : array-like, shape=[n_samples, dim_2, dim_3]
        ...
        an : array-like, shape=[n_samples, dim_n-1, dim_n]

        Returns
        -------
        mul : array-like, shape=[n_samples, dim_1, dim_n]
        """
        return reduce(gs.matmul, args)

    @classmethod
    def bracket(cls, mat_a, mat_b):
        """
        Return the commutator of a and b, i.e. `[a, b] = ab - ba`.

        Parameters
        ----------
        mat_a : array-like, shape=[n_samples, dim, dim]
        mat_b : array-like, shape=[n_samples, dim, dim]

        Returns
        -------
        mat_c : array-like, shape=[n_samples, dim, dim]
        """
        return cls.mul(mat_a, mat_b) - cls.mul(mat_b, mat_a)

    @staticmethod
    def transpose(mat):
        """Return the transpose of matrices.

        Parameters
        ----------
        mat : array-like, shape=[n_samples, dim, dim]

        Returns
        -------
        transpose : array-like, shape=[n_samples, dim, dim]
        """
        is_vectorized = (gs.ndim(gs.array(mat)) == 3)
        axes = (0, 2, 1) if is_vectorized else (1, 0)
        return gs.transpose(mat, axes)

    @classmethod
    def is_symmetric(cls, mat, atol=TOLERANCE):
        """
        Check if a matrix is symmetric.

        Parameters
        ----------
        mat : array-like, shape=[n_samples, n, n]
        atol : float, absolute tolerance. defaults to TOLERANCE

        Returns
        -------
        is_sym : array-like boolean, shape=[n_samples]
        """
        return cls.equal(mat, cls.transpose(mat), atol)

    @classmethod
    def make_symmetric(cls, mat):
        """
        Make a matrix symmetric, by averaging with its transpose.

        Parameters
        ----------
        mat : array-like, shape=[n_samples, n, n]

        Returns
        -------
        sym : array-like, shape=[n_samples, n, n]
        """
        return 1 / 2 * (mat + cls.transpose(mat))

    def random_uniform(self, n_samples=1):
        """Generate n samples from a uniform distribution."""
        point = gs.random.rand(n_samples, self.m, self.n)
        return point

    @classmethod
    def congruent(cls, mat_1, mat_2):
        """Compute the congruent action of mat_2 on mat_1.

        This is :math: `mat_2 mat_1 mat_2^T`.

        Parameters
        ----------
        mat_1 : array-like, shape=[n_samples, n, n]
        mat_2 : array-like, shape=[n_samples, n, n]

        Returns
        -------
        cong : array-like, shape=[n_samples, n, n]
        """
        return cls.mul(mat_2, mat_1, cls.transpose(mat_2))


class MatricesMetric(RiemannianMetric):
    """Euclidean metric on matrices given by Frobenius inner product."""

    def __init__(self, m, n):
        dimension = m * n
        super(MatricesMetric, self).__init__(
            dimension=dimension,
            signature=(dimension, 0, 0))

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute Frobenius inner product of two tan vecs at `base_point`."""
        tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=3)
        n_tangent_vecs_a, _, _ = tangent_vec_a.shape

        tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=3)
        n_tangent_vecs_b, _, _ = tangent_vec_b.shape

        assert n_tangent_vecs_a == n_tangent_vecs_b

        inner_prod = gs.einsum("nij,nij->n", tangent_vec_a, tangent_vec_b)
        inner_prod = gs.to_ndarray(inner_prod, to_ndim=1)
        inner_prod = gs.to_ndarray(inner_prod, to_ndim=2, axis=1)
        return inner_prod
