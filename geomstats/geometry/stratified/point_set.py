"""Class for Stratified Spaces.

Lead authors: Anna Calissano and Jonas Lueg
"""

from abc import ABC, abstractmethod

import geomstats.backend as gs


class Point(ABC):
    """Class for points of a set."""

    @abstractmethod
    def equal(self, point, atol=gs.atol):
        """Check equality against another point.

        Parameters
        ----------
        point : Point or list[Point]
            Point to compare against point.
        atol : float

        Returns
        -------
        is_equal : array-like, shape=[...]
        """


class PointCollection(ABC, list):
    """Class for point collections."""

    def equal(self, point, atol=gs.atol):
        """Check equality against another point.

        Parameters
        ----------
        point : Point or PointCollection
            Point to compare against point.
        atol : float
        """
        if isinstance(point, (list, tuple)):
            return gs.array(
                [
                    collection_point.equal(point_, atol)
                    for collection_point, point_ in zip(self, point)
                ]
            )

        return gs.array(
            [collection_point.equal(point, atol) for collection_point in self]
        )


class PointSet(ABC):
    r"""Class for a set of points of type Point."""

    def __init__(self, equip=True):
        if equip:
            self.equip_with_metric()

    def equip_with_metric(self, Metric=None, **metric_kwargs):
        """Equip manifold with Metric.

        Parameters
        ----------
        Metric : PointSetMetric object
            If None, default metric will be used.
        """
        if Metric is None:
            out = self.default_metric()
            if isinstance(out, tuple):
                Metric, kwargs = out
                kwargs.update(metric_kwargs)
                metric_kwargs = kwargs
            else:
                Metric = out

        self.metric = Metric(self, **metric_kwargs)

    @abstractmethod
    def belongs(self, point, atol=gs.atol):
        r"""Evaluate if a point belongs to the set.

        Parameters
        ----------
        point : Point or PointCollection
            Point to evaluate.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...]
            Boolean evaluating if point belongs to the set.
        """

    @abstractmethod
    def random_point(self, n_samples=1):
        r"""Sample random points on the PointSet.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        samples : Point or PointCollection
            Points sampled on the PointSet.
        """


class PointSetMetric(ABC):
    r"""Class for the lenght spaces.

    Parameters
    ----------
    space : PointSet
        Set to equip with metric.
    """

    def __init__(self, space):
        self._space = space

    @abstractmethod
    def dist(self, point_a, point_b):
        """Distance between two points in the PointSet.

        Parameters
        ----------
        point_a: Point or list[Point]
            Point in the PointSet.
        point_b: Point or list[Point]
            Point in the PointSet.

        Returns
        -------
        distance : array-like, shape=[...]
            Distance.
        """

    @abstractmethod
    def geodesic(self, initial_point, end_point):
        """Compute the geodesic in the PointSet.

        Parameters
        ----------
        initial_point: Point or list[Point]
            Point in the PointSet.
        end_point: Point or list[Point]
            Point in the PointSet.

        Returns
        -------
        path : callable
            Time parameterized geodesic curve.
        """
