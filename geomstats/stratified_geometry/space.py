"""Classes ``Point`` and ``Space`` for defining a set that is equipped with a metric."""

import abc

import geomstats.backend as gs
from geomstats.stratified_geometry.distance import Distance


class Point(abc.ABC):
    """Class for points of a metric space (set equipped with distance)."""

    def __init__(self, **kwargs):
        super(Point, self).__init__(**kwargs)

    @abc.abstractmethod
    def __repr__(self):
        """Produce a string of fancy representation of a point."""

    @abc.abstractmethod
    def __hash__(self):
        """Define a hash for the point."""


class Space(abc.ABC):
    """Class for the set of points of a metric space."""

    def __init__(self, point_type, dist: Distance, **kwargs):
        super(Space, self).__init__(**kwargs)
        self.point_type = point_type
        self.dist = dist

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
