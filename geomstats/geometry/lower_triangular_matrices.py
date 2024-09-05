"""The vector space of lower triangular matrices.

Lead author: Saiteja Utpala.
"""

import geomstats.backend as gs
from geomstats.geometry.base import LevelSet, MatrixVectorSpace
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


class StrictlyLowerTriangularMatrices(LevelSet, MatrixVectorSpace):
    r"""Strictly lower triangular matrices.

    Set of lower triangular matrices with null diagonal:

    .. math::

        \operatorname{LT}^0(n)=\{L \in \operatorname{LT}(n)
        \mid \operatorname{Diag}(L)=0\}

    Parameters
    ----------
    n : int
        Integer representing the shapes of the matrices: n x n.
    equip : bool
        If True, equip space with default metric.
    """

    def __init__(self, n, equip=True):
        self.n = n
        super().__init__(dim=int(n * (n - 1) / 2), equip=equip)

    def _define_embedding_space(self):
        """Define embedding space of the manifold.

        Returns
        -------
        embedding_space : Manifold
            Instance of Manifold.
        """
        return LowerTriangularMatrices(n=self.n)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return MatricesMetric

    def submersion(self, point):
        """Submersion that defines the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]

        Returns
        -------
        submersed_point : array-like, shape=[..., n]
        """
        return Matrices.diagonal(point)

    def tangent_submersion(self, vector, point):
        """Tangent submersion.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
        point : Ignored.

        Returns
        -------
        submersed_vector : array-like, shape=[..., n]
        """
        submersed_vector = Matrices.diagonal(vector)
        if point is not None and point.ndim > vector.ndim:
            return gs.broadcast_to(submersed_vector, point.shape[:-1])

        return submersed_vector

    def _create_basis(self):
        """Create a basis for the vector space.

        Returns
        -------
        basis : array-like, shape=[dim, n, n]
            Basis matrices of the space.
        """
        tril_idxs = gs.ravel_tril_indices(self.n, k=-1)
        vector_bases = gs.cast(
            gs.one_hot(tril_idxs, self.n * self.n),
            dtype=gs.get_default_dtype(),
        )
        print(vector_bases.shape)
        return gs.reshape(vector_bases, [-1, self.n, self.n])

    @staticmethod
    def basis_representation(matrix_representation):
        """Compute the coefficients of matrices in the given basis.

        Parameters
        ----------
        matrix_representation : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        vec : array-like, shape=[..., dim]
            Vector.
        """
        return gs.tril_to_vec(matrix_representation, k=-1)

    def projection(self, point):
        """Project a point to the manifold.

        Parameters
        ----------
        point: array-like, shape[..., n, n]
            Point to project.

        Returns
        -------
        proj_point: array-like, shape[..., n, n]
            Projected point.
        """
        proj_point = self.embedding_space.projection(point)
        return proj_point - gs.vec_to_diag(gs.diagonal(proj_point, axis1=-2, axis2=-1))
