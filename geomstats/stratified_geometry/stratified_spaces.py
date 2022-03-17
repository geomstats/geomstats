"""File for construction of sets of points and length spaces, i.e. metric spaces.

Lead authors: Anna Calissano & Jonas Lueg
"""

from abc import ABC, abstractmethod
from typing import TypeVar


def list_vectorize(fun):
    r"""Decoretor to vectorize the functions acting on Point as lists."""

    def wrapped(*args):
        if type(args[1]) is list:
            return fun(args[0], point=args[1])
        else:
            return fun(args[0], point=[args[1]])

    return wrapped


class Point(ABC):
    r"""Class for points of a set."""

    def __init__(self, **kwargs):
        super(Point, self).__init__(**kwargs)

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
        array_point : array-like, shape=[...,]
            An array representation of the Point type.
        """


P = TypeVar("P", bound=Point)


class PointSet(ABC):
    r"""Class for a set of points of type Point.

    Parameters
    ----------
    param: int
        Parameter defining the pointset.

    default_point_type : str, {\'vector\', \'matrix\'}
        Point type.
        Optional, default: 'Point'.

    default_coords_type : str, {\'intrinsic\', \'extrinsic\', etc}
        Coordinate type.
        Optional, default: 'intrinsic'.
    """

    def __init__(self):
        super(PointSet, self).__init__()

    @abstractmethod
    def belongs(self, point):
        r"""Evaluate if a point belongs to the set.

        Parameters
        ----------
        point : Point-like, shape=[..., dim]
            Point to evaluate.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the set.
        """

    @abstractmethod
    def random_point(self):
        r"""Sample random points on the PointSet.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        samples : Point-like List
            Points sampled on the PointSet.
        """

    @abstractmethod
    def set_to_array(self):
        """Covert a set of points into an array.

        Parameters
        ----------
        points : Point-like list, shape=[n, ...]
            Number of samples of point type to turn
            into an array.

        Returns
        -------
        points_array : array-like, shape=[n, ...]
            Points sampled on the hypersphere.
        """


class PointSetGeometry(PointSet, ABC):
    r"""Class for the lenght spaces.

    Parameters
    ----------
    Set : PointSet
        Underling PointSet.
    default_point_type : str, {\'vector\', \'matrix\'}
        Point type.
        Optional, default: \'PointType\'.
    default_coords_type : str, {\'intrinsic\', \'extrinsic\', etc}
        Coordinate type.
        Optional, default: \'intrinsic\'.
    """

    def __init__(self):
        super(PointSetGeometry, self).__init__()

    @abstractmethod
    def dist(self, point_a, point_b, **kwargs):
        """Distance between two points in the PointSet.

        Parameters
        ----------
        point_a: Point-like, shape=[..., ]
            Point in the PointSet.
        point_b: Point-like, shape=[..., ]
            Point in the PointSet.

        Returns
        -------
        distance : array-like, shape=[...,]
            Distance.
        """

    @abstractmethod
    def geodesic(self, point_a, point_b, **kwargs):
        """Compute the geodesic in the length space.

        Parameters
        ----------
        point_a: Point-like, shape=[..., ]
            Point in the PointSet.
        point_b: Point-like, shape=[..., ]
            Point in the PointSet.

        Returns
        -------
        geodesic : array-like, shape=[...,]
            Geodesic.
        """
