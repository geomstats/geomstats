"""Class for Stratified Spaces.

Lead authors: Anna Calissano & Jonas Lueg
"""

import functools
import itertools
from abc import ABC, abstractmethod

import geomstats.backend as gs


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
    if not isinstance(arg, (list, tuple)):
        return [arg], True

    return arg, False


def _manipulate_output_iterable(out):
    return PointCollection(out)


def _manipulate_output(
    out, to_list, manipulate_output_iterable=_manipulate_output_iterable
):
    is_array = gs.is_array(out)
    is_iterable = isinstance(out, (list, tuple))

    if not (gs.is_array(out) or is_iterable):
        return out

    if to_list:
        if is_array:
            return gs.array(out[0])
        if is_iterable:
            return out[0]

    if is_iterable:
        return manipulate_output_iterable(out)

    return out


def _vectorize_point(
    *args_positions,
    manipulate_input=_manipulate_input,
    manipulate_output=_manipulate_output,
):
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
            to_list = True
            args = list(args)
            for pos, name in args_positions:
                if name in kwargs:
                    kwargs[name], to_list_ = manipulate_input(kwargs[name])
                else:
                    args[pos], to_list_ = manipulate_input(args[pos])

                to_list = to_list and to_list_

            out = func(*args, **kwargs)

            return manipulate_output(out, to_list)

        return _wrapped

    return _dec


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
