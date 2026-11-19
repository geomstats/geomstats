"""Open intervals of the real line."""

import math

import geomstats.backend as gs
from geomstats.geometry.base import VectorSpaceOpenSet
from geomstats.geometry.euclidean import Euclidean, EuclideanMetric


class OpenInterval(VectorSpaceOpenSet):
    """Class for an open interval of the real line.

    The bounds may be infinite, e.g. for a half-line.

    Parameters
    ----------
    lower_bound : float
        Lower bound.
    upper_bound : float
        Upper bound.
    equip : bool
        If True, equip space with default metric.
    """

    def __init__(self, lower_bound=0.0, upper_bound=math.inf, equip=True):
        if not lower_bound < upper_bound:
            raise ValueError("The lower bound must be smaller than the upper bound.")

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        super().__init__(
            dim=1,
            embedding_space=Euclidean(dim=1, equip=False),
            equip=equip,
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return OpenIntervalMetric

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the open interval.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point to be checked.
        atol : float
            Tolerance.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the open interval.
        """
        point = point[..., 0]
        return gs.logical_and(
            self.lower_bound - atol < point, point < self.upper_bound + atol
        )

    def projection(self, point, atol=gs.atol):
        """Project a point in ambient space to the open interval.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point in ambient space.
        atol : float
            Distance to the bounds of the projected points.

        Returns
        -------
        projected : array-like, shape=[..., 1]
            Projected point.
        """
        return gs.clip(point, self.lower_bound + atol, self.upper_bound - atol)

    def random_point(self, n_samples=1, bound=1.0, low=None):
        """Sample points in the open interval.

        The uniform distribution on [low, low + bound) is used, clipped to the
        upper bound of the interval. `low` defaults to the lower bound of the
        interval if finite, and otherwise to the upper bound minus `bound`, or
        to `-bound / 2` for the real line.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        bound : float
            Length of the sampling interval.
        low : float
            Lower end of the sampling interval.

        Returns
        -------
        samples : array-like, shape=[..., 1]
            Points sampled in the open interval.
        """
        if low is None:
            if not math.isinf(self.lower_bound):
                low = self.lower_bound + gs.atol
            elif not math.isinf(self.upper_bound):
                low = self.upper_bound - bound
            else:
                low = -bound / 2

        high = min(low + bound, self.upper_bound)

        size = (n_samples, 1) if n_samples != 1 else (1,)
        return gs.random.uniform(low=low, high=high, size=size)


class OpenIntervalMetric(EuclideanMetric):
    """Class for the Euclidean metric on an open interval."""

    def injectivity_radius(self, base_point):
        """Compute the radius of the injectivity domain.

        It is the distance to the closest bound.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        point = base_point[..., 0]
        return gs.minimum(
            point - self._space.lower_bound, self._space.upper_bound - point
        )
