"""Class for Stratified Spaces.

Lead authors: Anna Calissano & Jonas Lueg
"""

from abc import ABC, abstractmethod


def _belongs_vectorize(fun):
    r"""Vectorize the belongs acting on Point as lists."""

    def wrapped(*args, **kwargs):
        r"""Vectorize the belongs."""
        if type(args[1]) is not list:
            args = list(args)
            args[1] = [args[1]]

        return fun(*args, **kwargs)

    return wrapped


def _dist_vectorize(fun):
    r"""Vectorize the distance acting on Point as lists."""

    def wrapped(*args, **kwargs):
        r"""Vectorize the distance."""
        args = list(args)
        if type(args[1]) is list and type(args[2]) is list:
            pass
        elif type(args[1]) is not list and type(args[2]) is not list:
            args[1], args[2] = [args[1]], [args[2]]
        elif type(args[1]) is not list:
            args[1] = [args[1]]
        else:
            args[2] = [args[2]]

        return fun(*args, **kwargs)

    return wrapped


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
    r"""Class for a set of points of type Point.

    Parameters
    ----------
    param: int
        Parameter defining the pointset.

    default_point_type : str, {\'vector\', \'matrix\', \'Point\'}
        Point type.
        Optional, default: \'Point\'.

    default_coords_type : str, {\'intrinsic\', \'extrinsic\', etc}
        Coordinate type.
        Optional, default: \'intrinsic\'.
    """

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


class PointSetGeometry(ABC):
    r"""Class for the lenght spaces.

    Parameters
    ----------
    Set : PointSet
        Underling PointSet.
    default_point_type : str, {\'vector\', \'matrix\', \'Point\' }
        Point type.
        Optional, default: \'Point\'.
    default_coords_type : str, {\'intrinsic\', \'extrinsic\', etc}
        Coordinate type.
        Optional, default: \'intrinsic\'.
    """

    def __init__(self, space: PointSet, **kwargs):
        super(PointSetGeometry, self).__init__(**kwargs)
        self.space = space

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
    def geodesic(self, point_a, point_b, **kwargs):
        """Compute the geodesic in the PointSet.

        Parameters
        ----------
        point_a: Point or List of Points, shape=[...]
            Point in the PointSet.
        point_b: Point or List of Points, shape=[...]
            Point in the PointSet.

        Returns
        -------
        path : callable
            Time parameterized geodesic curve.
        """
