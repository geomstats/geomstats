"""Module providing the SkewSymmetricMatrices class.

This is the Lie algebra of the Special Orthogonal Group.
As basis we choose the matrices with a single 1 on the upper triangular part
of the matrices (and a -1 in its lower triangular part), except in dim 2 and
3 to match usual conventions.
"""

import geomstats.backend as gs
from geomstats.geometry.lie_algebra import MatrixLieAlgebra


TOLERANCE = 1e-8


class SkewSymmetricMatrices(MatrixLieAlgebra):
    """Class for skew-symmetric matrices.

    Parameters
    ----------
    n : int
        Number of rows and columns.
    """

    def __init__(self, n):
        dim = int(n * (n - 1) / 2)
        super(SkewSymmetricMatrices, self).__init__(dim, n)

        if n == 2:
            self.basis = gs.array([[[0., -1.], [1., 0.]]])
        elif n == 3:
            self.basis = gs.array([
                [[0., 0., 0.],
                 [0., 0., -1.],
                 [0., 1., 0.]],
                [[0., 0., 1.],
                 [0., 0., 0.],
                 [-1., 0., 0.]],
                [[0., -1., 0.],
                 [1., 0., 0.],
                 [0., 0., 0.]]])
        else:
            self.basis = gs.zeros((dim, n, n))
            basis = []
            for row in gs.arange(n - 1):
                for col in gs.arange(row + 1, n):
                    basis.append(gs.array_from_sparse(
                        [(row, col), (col, row)], [1., -1.], (n, n)))
            self.basis = gs.stack(basis)

    def belongs(self, mat, atol=TOLERANCE):
        """Evaluate if mat is a skew-symmetric matrix.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Square matrix to check.
        atol : float
            Tolerance for the equality evaluation.
            Optional, default: TOLERANCE.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if matrix is skew symmetric.
        """
        is_skew = self.is_skew_symmetric(mat=mat, atol=atol)
        return gs.logical_and(
            is_skew, super(SkewSymmetricMatrices, self).belongs(mat))

    @classmethod
    def projection(cls, mat):
        r"""Compute the skew-symmetric component of a matrix.

        The skew-symmetric part of a matrix :math: `X` is defined by
        .. math:
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
        return cls.to_skew_symmetric(mat)

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
            vec = gs.stack([
                matrix_representation[..., 2, 1],
                matrix_representation[..., 0, 2],
                matrix_representation[..., 1, 0]])
            return gs.transpose(vec)

        return gs.triu_to_vec(matrix_representation, k=1)
