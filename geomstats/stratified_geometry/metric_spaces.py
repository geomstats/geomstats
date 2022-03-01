"""File for construction of sets of points and length spaces, i.e. metric spaces.

Lead authors: Anna Calissano & Jonas Lueg
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar


class Point(ABC):
    """Class for points of a set."""

    def __init__(self, **kwargs):
        super(Point, self).__init__(**kwargs)

    @abstractmethod
    def __repr__(self):
        """Produce a string of fancy representation of a point."""

    @abstractmethod
    def __hash__(self):
        """Define a hash for the point."""


P = TypeVar("P", bound=Point)


class PointSet(Generic[P], ABC):
    """Class for a set of points, where points are instances of a class ``P``."""

    def __init__(self):
        super(PointSet, self).__init__()

    def belongs(self, point):
        """Check whether a point belongs to the space."""
        return type(point) is P

    @abstractmethod
    def random_point(self) -> P:
        """Compute a random point in the space."""


class LengthSpace(PointSet[P], ABC):
    """Class for a length space, i.e. a metric space."""

    def __init__(self):
        super(LengthSpace, self).__init__()

    @abstractmethod
    def dist(self, p: P, q: P, **kwargs):
        """Compute the distance between two points."""

    @abstractmethod
    def geodesic(self, p: P, q: P, **kwargs):
        """Compute the geodesic (in metric space sense) between two points."""
