"""Manifold module.

In other words, a topological space that locally resembles
Euclidean space near each point.

Lead author: Nina Miolane.
"""

import abc

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.riemannian_metric import RiemannianMetric


class Manifold(abc.ABC):
    r"""Class for manifolds.

    Parameters
    ----------
    dim : int
        Dimension of the manifold.
    shape : tuple of int
        Shape of one element of the manifold.
        Optional, default : None.
    metric : RiemannianMetric
        Metric object to use on the manifold.
    default_coords_type : str, {\'intrinsic\', \'extrinsic\', etc}
        Coordinate type.
        Optional, default: 'intrinsic'.
    """

    def __init__(
        self, dim, shape, metric=None, default_coords_type="intrinsic", **kwargs
    ):
        super().__init__(**kwargs)
        geomstats.errors.check_integer(dim, "dim")

        if not isinstance(shape, tuple):
            raise ValueError("Expected a tuple for the shape argument.")

        self.dim = dim
        self.shape = shape
        self.default_coords_type = default_coords_type
        self._metric = metric

    @property
    def default_point_type(self):
        """Point type.

        `vector` or `matrix`.
        """
        if len(self.shape) == 1:
            return "vector"
        return "matrix"

    @abc.abstractmethod
    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to evaluate.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the manifold.
        """

    @abc.abstractmethod
    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """

    @abc.abstractmethod
    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """

    @abc.abstractmethod
    def random_point(self, n_samples=1, bound=1.0):
        """Sample random points on the manifold.

        If the manifold is compact, a uniform distribution is used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample for non compact manifolds.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., {dim, [n, n]}]
            Points sampled on the manifold.
        """

    def regularize(self, point):
        """Regularize a point to the canonical representation for the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point.

        Returns
        -------
        regularized_point : array-like, shape=[..., dim]
            Regularized point.
        """
        regularized_point = point
        return regularized_point

    @property
    def metric(self):
        """Riemannian Metric associated to the Manifold."""
        return self._metric

    @metric.setter
    def metric(self, metric):
        if metric is not None:
            if not isinstance(metric, RiemannianMetric):
                raise ValueError("The argument must be a RiemannianMetric object")
            if metric.dim != self.dim:
                metric.dim = self.dim
        self._metric = metric

    def random_tangent_vec(self, base_point, n_samples=1):
        """Generate random tangent vec.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        base_point :  array-like, shape=[..., dim]
            Point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vec at base point.
        """
        if (
            n_samples > 1
            and base_point.ndim > len(self.shape)
            and n_samples != len(base_point)
        ):
            raise ValueError(
                "The number of base points must be the same as the "
                "number of samples, when different from 1."
            )
        return gs.squeeze(
            self.to_tangent(
                gs.random.normal(size=(n_samples,) + self.shape), base_point
            )
        )
