"""
Base class for all manifolds.
The OOP structure has been inspired by Andrea Censi, geometry module.
"""

from abc import abstractmethod


class Manifold(object):
    """Base class for differentiable manifolds."""

    def __init__(self, dimension):
        self.dimension = dimension

    @abstractmethod
    def belongs(self, point):
        """Check if the point belongs to the manifold."""

    def regularize(self, point):
        """
        Regularizes the point's coordinates to the canonical representation
        for this manifold.
        """
        return point

    def friendly(self, point):
        """
        Returns a friendly description string for a point on the manifold.
        """
        return point.__str__()


class RiemannianManifold(Manifold):
    """ Base class for (pseudo-/sub-) Riemannian manifolds."""

    def __init__(self, dimension, metric):
        self.dimension = dimension
        self.metric = metric

    @abstractmethod
    def riemannian_exp(self, ref_point, tangent_vec):
        """
        Compute the Riemannian exponential at point ref_point
        of tangent vector tangent_vec wrt the metric.
        """

    @abstractmethod
    def riemannian_log(self, ref_point, tangent_vec):
        """
        Compute the Riemannian logarithm at point ref_point
        of tangent vector tangent_vec wrt the metric.
        """

    @abstractmethod
    def riemannian_dist(point_a, point_b):
        """
        Compute the Riemannian distance between points
        point_a and point_b.
        """

    @abstractmethod
    def random_uniform(self):
        """
        Samples a random point in this manifold according to
        the Riemannian measure.
        """


class LieGroup(Manifold):
    """ Base class for Lie groups."""

    def __init__(self, dimension):
        self.dimension = dimension
        self.identity
