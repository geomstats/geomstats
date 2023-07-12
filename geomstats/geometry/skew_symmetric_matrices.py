"""Module providing the SkewSymmetricMatrices class.

This is the Lie algebra of the Special Orthogonal Group.
As basis we choose the matrices with a single 1 on the upper triangular part
of the matrices (and a -1 in its lower triangular part), except in dim 2 and
3 to match usual conventions.

Lead author: Nicolas Guigui.
"""

import geomstats.backend as gs
from geomstats.geometry.lie_algebra import MatrixLieAlgebra
from geomstats.geometry.matrices import Matrices


class SkewSymmetricMatrices(MatrixLieAlgebra):
    """Class for skew-symmetric matrices.

    Parameters
    ----------
    n : int
        Number of rows and columns.
    """

    def __init__(self, n):
        self.n = n
        dim = int(n * (n - 1) / 2)
        super().__init__(dim=dim, representation_dim=n, equip=False)
        self.embedding_space = Matrices(n, n)

    def _create_basis(self):
        """Create the canonical basis."""
        n = self.n
        if n == 2:
            return gs.array([[[0.0, -1.0], [1.0, 0.0]]])
        if n == 3:
            return gs.array(
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ]
            )
        indices, data = [], []
        k = -1
        for row in range(n - 1):
            for col in range(row + 1, n):
                k += 1
                indices.extend([(k, row, col), (k, col, row)])
                data.extend([1.0, -1.0])

        return gs.array_from_sparse(indices, data, (k + 1, n, n))

    def belongs(self, point, atol=gs.atol):
        """Evaluate if point is a skew-symmetric matrix.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Square matrix to check.
        atol : float
            Tolerance for the equality evaluation.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if matrix is skew symmetric.
        """
        has_right_shape = self.embedding_space.belongs(point)
        if gs.all(has_right_shape):
            return Matrices.is_skew_symmetric(mat=point, atol=atol)
        return has_right_shape

    def random_point(self, n_samples=1, bound=1.0):
        """Sample from a uniform distribution in a cube and project to skew-symmetric.

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
        point : array-like, shape=[..., n, n]
            Sample.
        """
        return self.projection(super().random_point(n_samples, bound))

    @classmethod
    def projection(cls, mat):
        r"""Compute the skew-symmetric component of a matrix.

        The skew-symmetric part of a matrix :math:`X` is defined by

        .. math::
            (X - X^T) / 2

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        skew_sym : array-like, shape=[..., n, n]
            Skew-symmetric matrix.
        """
        return Matrices.to_skew_symmetric(mat)

    def basis_representation(self, matrix_representation):
        """Calculate the coefficients of given matrix in the basis.

        Compute a 1d-array that corresponds to the input matrix in the basis
        representation.

        Parameters
        ----------
        matrix_representation : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        basis_representation : array-like, shape=[..., dim]
            Representation in the basis.
        """
        if self.n == 2:
            return matrix_representation[..., 1, 0][..., None]
        if self.n == 3:
            vec = gs.stack(
                [
                    matrix_representation[..., 2, 1],
                    matrix_representation[..., 0, 2],
                    matrix_representation[..., 1, 0],
                ]
            )
            return gs.transpose(vec)

        return gs.triu_to_vec(matrix_representation, k=1)
