"""The vector space of lower triangular matrices.

Lead author: Saiteja Utpala.
"""

import geomstats.backend as gs
from geomstats.geometry.base import VectorSpace
from geomstats.geometry.matrices import Matrices, MatricesMetric


class LowerTriangularMatrices(VectorSpace):
    """Class for the vector space of lower triangular matrices of size n.

    Parameters
    ----------
    n : int
        Integer representing the shapes of the matrices: n x n.
    """

    def __init__(self, n, **kwargs):
        super(LowerTriangularMatrices, self).__init__(
            dim=int(n * (n + 1) / 2),
            shape=(n, n),
            metric=MatricesMetric(n, n),
            default_point_type="matrix",
        )
        self.n = n

    def _create_basis(self):
        """Compute the basis of the vector space of lower triangular.

        Returns
        -------
        basis : array-like, shape=[dim, n, n]
            Basis matrices of the space.
        """
        tril_idxs = gs.ravel_tril_indices(self.n)
        vector_bases = gs.one_hot(tril_idxs, self.n * self.n)
        return gs.reshape(vector_bases, [-1, self.n, self.n])

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a matrix is lower triangular.

        Parameters
        ----------
        point : array-like, shape=[.., n, n]
            Point to test.
        atol : float
            Tolerance to evaluate equality with the transpose.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the space.
        """
        belongs = super(LowerTriangularMatrices, self).belongs(point)
        if gs.any(belongs):
            is_lower_triangular = Matrices.is_lower_triangular(point, atol)
            return gs.logical_and(belongs, is_lower_triangular)
        return belongs

    @staticmethod
    def to_vector(mat):
        """Convert a lower triangular matrix into a vector.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        vec : array-like, shape=[..., n(n+1)/2]
            Vector.
        """
        return gs.tril_to_vec(mat)

    def projection(self, point):
        """Make a square matrix lower triangular by zeroing out other elements.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        sym : array-like, shape=[..., n, n]
            Symmetric matrix.
        """
        return Matrices.to_lower_triangular(point)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample a lower triangular matrix with a uniform distribution in a box.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Side of hypercube support of the uniform distribution.
            Optional, default: 1.0.

        Returns
        -------
        point : array-like, shape=[..., n, n]
           Sample.
        """
        sample = super(LowerTriangularMatrices, self).random_point(n_samples, bound)
        return Matrices.to_lower_triangular(sample)
