"""Class for the group of special linear matrices."""

import geomstats.backend as gs
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.lie_group import MatrixLieGroup
from geomstats.geometry.lie_algebra import MatrixLieAlgebra


class SpecialLinearMatrices(MatrixLieGroup):
    
    def __init__(self, n):
        super(SpecialLinearMatrices, self).__init__(
            dim=int((n * (n - 1)) / 2),
            n=n,
            lie_algebra=SpecialLinearMatrices(n=n),
        )

        self.metric = InvariantMetric(self)

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the group.

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
        pass

    def projection(self, point):
        """Project a point in embedding space to the group.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., n, n]
            Projected point.
        """
        pass

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
        pass


class SpecialLinearMatrixLieAlgebra(MatrixLieAlgebra):

    def __init__(self, n):
        super(SpecialLinearMatrixLieAlgebra, self).__init__(
            dim=int((n * (n - 1)) / 2),
            n=n,
        )

    def basis_representation(self, matrix_representation):
        """Compute the coefficients of matrices in the given basis.

        Parameters
        ----------
        matrix_representation : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        basis_representation : array-like, shape=[..., dim]
            Coefficients in the basis.
        """
        pass

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
        pass

    def projection(self, point):
        """Project a point to the Lie algebra.

        Parameters
        ----------
        point: array-like, shape=[..., n, n]
            Point.

        Returns
        -------
        point: array-like, shape=[..., n, n]
            Projected point.
        """
        pass
