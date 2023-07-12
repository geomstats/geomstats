"""Class for Stratified Spaces.

Lead authors: Anna Calissano & Jonas Lueg
"""

import functools
import itertools
from abc import ABC, abstractmethod


def broadcast_lists(list_a, list_b):
    """Broadcast two lists.

    Similar behavior as ``gs.broadcast_arrays``, but for lists.
    """
    n_a = len(list_a)
    n_b = len(list_b)

    if n_a == n_b:
        return list_a, list_b

    if n_a == 1:
        return itertools.zip_longest(list_a, list_b, fillvalue=list_a[0])

    if n_b == 1:
        return itertools.zip_longest(list_a, list_b, fillvalue=list_b[0])

    raise Exception(f"Cannot broadcast lens {n_a} and {n_b}")


def _manipulate_input(arg):
    if not (type(arg) in [list, tuple]):
        return [arg]

    return arg


def _vectorize_point(*args_positions, manipulate_input=_manipulate_input):
    """Check point type and transform in iterable if not the case.

    Parameters
    ----------
    args_positions : tuple
        Position and corresponding argument name. A tuple for each position.

    Notes
    -----
    Explicitly defining args_positions and args names ensures it works for all
    combinations of input calling.
    """

    def _dec(func):
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            args = list(args)
            for pos, name in args_positions:
                if name in kwargs:
                    kwargs[name] = manipulate_input(kwargs[name])
                else:
                    args[pos] = manipulate_input(args[pos])

            return func(*args, **kwargs)

        return _wrapped

    return _dec


class Point(ABC):
    r"""Class for points of a set."""

    @abstractmethod
    def __repr__(self):
        """Produce a string with a verbal description of the point."""

    @abstractmethod
    def __hash__(self):
        """Define a hash for the point."""

    @abstractmethod
    def to_array(self):
        """Turn the point into a numpy array.

        Returns
        -------
        array_point : array-like, shape=[...]
            An array representation of the Point type.
        """


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
    def belongs(self, point, atol):
        r"""Evaluate if a point belongs to the set.

        Parameters
        ----------
        point : Point-like, shape=[...]
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
        samples : List of Point
            Points sampled on the PointSet.
        """

    @abstractmethod
    def set_to_array(self, points):
        """Convert a set of points into an array.

        Parameters
        ----------
        points : list of Point, shape=[...]
            Number of samples of point type to turn
            into an array.

        Returns
        -------
        points_array : array-like, shape=[...]
            Points sampled on the PointSet.
        """


class PointSetMetric(ABC):
    r"""Class for the lenght spaces."""

    def __init__(self, space):
        self._space = space

    @abstractmethod
    def dist(self, point_a, point_b, **kwargs):
        """Distance between two points in the PointSet.

        Parameters
        ----------
        point_a: Point or List of Point, shape=[...]
            Point in the PointSet.
        point_b: Point or List of Point, shape=[...]
            Point in the PointSet.

        Returns
        -------
        distance : array-like, shape=[...]
            Distance.
        """

    @abstractmethod
    def geodesic(self, initial_point, end_point, **kwargs):
        """Compute the geodesic in the PointSet.

        Parameters
        ----------
        initial_point: Point or List of Points, shape=[...]
            Point in the PointSet.
        end_point: Point or List of Points, shape=[...]
            Point in the PointSet.

        Returns
        -------
        path : callable
            Time parameterized geodesic curve.
        """
