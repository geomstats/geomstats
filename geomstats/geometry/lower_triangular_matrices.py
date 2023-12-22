"""The vector space of lower triangular matrices.

Lead author: Saiteja Utpala.
"""

import geomstats.backend as gs
from geomstats.geometry.base import MatrixVectorSpace
from geomstats.geometry.matrices import Matrices, MatricesMetric


class LowerTriangularMatrices(MatrixVectorSpace):
    """Class for the vector space of lower triangular matrices of size n.

    Parameters
    ----------
    n : int
        Integer representing the shapes of the matrices: n x n.
    """

    def __init__(self, n, equip=True):
        super().__init__(dim=int(n * (n + 1) / 2), shape=(n, n), equip=equip)
        self.n = n

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return MatricesMetric

    def _create_basis(self):
        """Compute the basis of the vector space of lower triangular.

        Returns
        -------
        basis : array-like, shape=[dim, n, n]
            Basis matrices of the space.
        """
        tril_idxs = gs.ravel_tril_indices(self.n)
        vector_bases = gs.cast(
            gs.one_hot(tril_idxs, self.n * self.n),
            dtype=gs.get_default_dtype(),
        )
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
        belongs = super().belongs(point)
        if gs.any(belongs):
            is_lower_triangular = Matrices.is_lower_triangular(point, atol)
            return gs.logical_and(belongs, is_lower_triangular)
        return belongs

    @staticmethod
    def basis_representation(matrix_representation):
        """Convert a lower triangular matrix into a vector.

        Parameters
        ----------
        matrix_representation : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        vec : array-like, shape=[..., n(n+1)/2]
            Vector.
        """
        return gs.tril_to_vec(matrix_representation)

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
        return gs.tril(point)
