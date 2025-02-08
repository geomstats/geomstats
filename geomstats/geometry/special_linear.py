"""Class for the group of special linear matrices."""

import geomstats.algebra_utils as utils
import geomstats.backend as gs
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.lie_algebra import MatrixLieAlgebra
from geomstats.geometry.lie_group import MatrixLieGroup
from geomstats.geometry.matrices import Matrices


class SpecialLinear(MatrixLieGroup):
    """Class for the Special Linear group SL(n).

    This is the space of invertible matrices of size n and unit determinant.
    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n):
        """Construct an instance of SpecialLinear()."""
        super(SpecialLinear, self).__init__(
            dim=int((n * (n - 1)) / 2),
            representation_dim=n,
            lie_algebra=TracelessMatrices(n=n),
        )
        self.metric = InvariantMetric(self)

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the group.

        Check the size and the value of the determinant.
        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point to evaluate.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.
        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the manifold.
        """
        has_right_shape = self.lie_algebra.ambient_space.belongs(point)

        if gs.all(has_right_shape):
            return gs.isclose(gs.linalg.det(point), 1.0, atol=atol)

        return has_right_shape

    def projection(self, point):
        """Project a point in embedding space to the group.

        This can be done by scaling the entire matrix by its determinant to
        the power 1/n.
        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point in embedding manifold.
        Returns
        -------
        projected : array-like, shape=[..., n, n]
            Projected point.
        """
        proj_point = utils.flip_determinant(gs.copy(point),
                                            gs.linalg.det(point))
        scale_coeff = gs.power(gs.linalg.det(proj_point),
                               1.0 / self.representation_dim)
        return gs.einsum("...ij,...->...ij", proj_point, 1.0 / scale_coeff)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in the group.

        One may use a sample from the general linear group and project it
        down to the special linear group.
        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Side of hypercube support of the uniform distribution.
            Optional, default: 1.0
        Returns
        -------
        point : array-like, shape=[..., dim]
           Sample.
        """
        sample = self.lie_algebra.ambient_space.random_point(
            n_samples=n_samples, bound=bound
        )
        return self.projection(sample)


class TracelessMatrices(MatrixLieAlgebra):
    """Class for the Lie algebra sl(n) of the Special Linear group.

    This is the space of matrices of size n with vanishing trace.
    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n):
        """Construct an instance of TracelessMatrices()."""
        super(TracelessMatrices, self).__init__(
            dim=n**2 - 1,
            representation_dim=n,
        )
        self.ambient_space = Matrices(n, n)

    def _get_basis_nondiag_indices(self):
        return gs.array(
            [
                i != j
                for j in range(self.representation_dim)
                for i in range(self.representation_dim)
            ]
        )

    def _create_basis(self):
        indices = self._get_basis_nondiag_indices()
        basis = gs.reshape(
            gs.eye(self.representation_dim * self.representation_dim)[indices],
            (
                self.representation_dim**2 - self.representation_dim,
                self.representation_dim,
                self.representation_dim,
            ),
        )
        diag_basis = []
        diag_rows, diag_cols = gs.diag_indices(self.representation_dim)
        for row, col in zip(diag_rows[1:], diag_cols[1:]):
            diag_basis.append(
                gs.array_from_sparse(
                    [(0, 0), (row, col)],
                    [-1.0, 1.0],
                    (self.representation_dim, self.representation_dim),
                )
            )

        return gs.concatenate([basis, gs.stack(diag_basis)])

    def basis_representation(self, matrix_representation):
        """Compute the coefficients of matrices in the given basis.

        Assume the basis is the one described in this answer on StackOverflow:
        https://math.stackexchange.com/a/1949395/920836
        Parameters
        ----------
        matrix_representation : array-like, shape=[..., n, n]
            Matrix.
        Returns
        -------
        basis_representation : array-like, shape=[..., dim]
            Coefficients in the basis.
        """
        indices = gs.reshape(
            self._get_basis_nondiag_indices(),
            (self.representation_dim, self.representation_dim),
        )

        diag_rows, diag_cols = gs.diag_indices(self.representation_dim)

        vec = gs.concatenate(
            [
                matrix_representation[..., indices],
                matrix_representation[..., diag_rows[1:], diag_cols[1:]],
            ],
            axis=-1,
        )

        return vec

    def belongs(self, point, atol=gs.atol):
        """Evaluate if the point belongs to the Lie algebra.

        This method checks the shape of the input point and its trace.
        Parameters
        ----------
        point : array-like, shape=[.., n, n]
            Point to test.
        atol : float
            Tolerance threshold for zero values.
        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the space.
        """
        has_right_shape = self.ambient_space.belongs(point)
        if gs.all(has_right_shape):
            return gs.isclose(gs.trace(point), 0.0, atol=atol)

        return has_right_shape

    def projection(self, point):
        """Project a point to the Lie algebra.

        This can be done by removing the trace in the first entry of the
        matrix.
        Parameters
        ----------
        point: array-like, shape=[..., n, n]
            Point.
        Returns
        -------
        point: array-like, shape=[..., n, n]
            Projected point.
        """
        trace = gs.trace(point)
        sps_matrix = gs.array_from_sparse(
            [(0, 0)], [1.0], (self.representation_dim, self.representation_dim)
        )

        return gs.copy(point) - gs.einsum("...,...ij->...ij",
                                          trace, sps_matrix)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in the group.

        One may use a sample from the general linear group and project it
        down to the special linear group.
        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Side of hypercube support of the uniform distribution.
            Optional, default: 1.0
        Returns
        -------
        point : array-like, shape=[..., dim]
           Sample.
        """
        sample = self.ambient_space.random_point(n_samples=n_samples,
                                                 bound=bound)
        return self.projection(sample)
