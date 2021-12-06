"""Class for the group of special linear matrices."""

import geomstats.backend as gs
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.lie_group import MatrixLieGroup
from geomstats.geometry.lie_algebra import MatrixLieAlgebra
from geomstats.geometry.general_linear import GeneralLinear


class SpecialLinear(MatrixLieGroup):
    """Class for the Special Linear group SL(n).

    This is the space of invertible matrices of size n and unit determinant.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n):
        super(SpecialLinear, self).__init__(
            dim=int((n * (n - 1)) / 2),
            n=n,
            lie_algebra=SpecialLinearLieAlgebra(n=n),
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
            # ???: det can be negative?
            return gs.isclose(gs.abs(gs.linalg.det(point)), 1.0, atol=atol)

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
        scale_coeff = gs.power(gs.abs(gs.linalg.det(point)),
                               1.0 / self.n)
        aux_shape = [1] if len(gs.shape(point)) == 2 else [1, 1]
        return gs.divide(point, gs.reshape(scale_coeff, (-1, *aux_shape)))

    def random_point(self, n_samples=1, bound=1.0, n_iter=100):
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
        n_iter : int
            Maximum number of trials to sample a matrix with non zero det.
            Optional, default: 100.

        Returns
        -------
        point : array-like, shape=[..., dim]
           Sample.
        """
        sample = self.lie_algebra.ambient_space.random_point(
            n_samples=n_samples, bound=bound, n_iter=n_iter)
        return self.projection(sample)


class SpecialLinearLieAlgebra(MatrixLieAlgebra):
    """Class for the Lie algebra sl(n) of the Special Linear group.

    This is the space of matrices of size n with vanishing trace.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n):
        super(SpecialLinearLieAlgebra, self).__init__(
            dim=n**2 - 1,
            n=n,
        )
        # ???: not sure it is in the right place
        self.ambient_space = GeneralLinear(n)

        # create basis
        rows, cols = gs.triu_indices(self.n, k=1)
        diag_rows, diag_cols = gs.diag_indices(self.n)

        dims = gs.arange(0, self.dim)
        n_non_diag_half = (self.dim - n + 1) // 2

        self.basis = gs.zeros((self.dim, n, n))

        self.basis[dims[:n_non_diag_half], rows, cols] = 1.
        self.basis[dims[n_non_diag_half:-n + 1], cols, rows] = 1.

        self.basis[dims[-n + 1:], 0, 0] = -1
        self.basis[dims[-n + 1:], diag_rows[1:], diag_cols[1:]] = 1

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
        rows, cols = gs.triu_indices(self.n, k=1)
        diag_rows, diag_cols = gs.diag_indices(self.n)

        vec = gs.concatenate(
            [
                matrix_representation[..., rows, cols],
                matrix_representation[..., cols, rows],
                matrix_representation[..., diag_rows[1:], diag_cols[1:]],
            ], axis=-1
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
            return gs.isclose(gs.trace(point, axis1=-2, axis2=-1),
                              0.0, atol=atol)

        return has_right_shape

    def projection(self, point):
        """Project a point to the Lie algebra.

        This can be done by removing the trace in the first entry of the matrix.

        Parameters
        ----------
        point: array-like, shape=[..., n, n]
            Point.

        Returns
        -------
        point: array-like, shape=[..., n, n]
            Projected point.
        """
        projected_point = gs.copy(point)
        projected_point[..., 0, 0] = projected_point[..., 0, 0] - gs.trace(
            point, axis1=-2, axis2=-1)

        return projected_point
