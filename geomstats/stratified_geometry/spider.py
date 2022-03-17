"""Class for the tripod or three-spider.

Lead authors: Anna Calissano & Jonas Lueg
"""

import geomstats.backend as gs
import geomstats.stratified_geometry.stratified_spaces
from geomstats.geometry.euclidean import Euclidean
from geomstats.stratified_geometry.stratified_spaces import (
    Point,
    PointSet,
    PointSetGeometry,
)


class SpiderPoint(Point):
    r"""Class for points of the Spider.

    Parameters
    ----------
    s : int
        The stratum, an integer indicating the stratum the point lies in. If zero, then
        the point is on the origin.
    x : float
        A positive number, the coordinate of the point. It must be zero if and only if
        the stratum is zero, i.e. the origin.
    """

    def __init__(self, s, x):
        super(Point, self).__init__()
        if s == 0 and x != 0:
            raise ValueError("If the stratum is zero, x must be zero.")
        self.s = s
        self.x = x

    def __repr__(self):
        """Return a readable representation of the instance."""
        return f"s{self.s}: {gs.round(self.x, 5)}"

    def __hash__(self):
        """Return the hash of the instance."""
        return hash((self.s, self.x))

    def to_array(self):
        """Return the hash of the instance."""
        return gs.array([self.s, self.x])


class Spider(PointSet):
    r"""Spider: a set of rays attached to the origin.

    Parameters
    ----------
    rays : int
        Number of rays to attach to the origin.
        Note that zero counts as the origin not as a ray.
    """

    def __init__(self, rays):
        super(Spider, self).__init__()
        self.rays = rays

    def random_point(self, n_samples=1):
        r"""Compute a random point of the spider set.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        samples : SpiderPoint-like, shape=[...,n_sample]
            List of SpiderPoints.
        """
        if self.rays != 0:
            s = gs.random.randint(low=0, high=self.rays, size=n_samples)
            x = gs.abs(gs.random.normal(loc=10, scale=1, size=n_samples))
            x[s == 0] = 0
            return [SpiderPoint(s=s[k], x=x[k]) for k in range(n_samples)]
        else:
            return [SpiderPoint(s=0, x=0)] * n_samples

    @geomstats.stratified_geometry.stratified_spaces.list_vectorize
    def belongs(self, point):
        r"""Check if a random point belongs to the spider set.

        Parameters
        ----------
        point : SpiderPoint
             Point to be checked.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if the SpiderPoint belongs to the Spider Set.
        """
        results = []
        for single_point in point:
            results += [
                self.value_check(single_point)
                and self.rays_check(single_point)
                and self.zero_check(single_point)
                and type(single_point) == SpiderPoint
            ]
        return gs.array(results)

    def rays_check(self, single_point):
        r"""Check if a random point has the correct number of rays.

        Parameters
        ----------
        point : SpiderPoint
             Point to be checked.

        Returns
        -------
        belongs : boolean
            Boolean denoting if the point has a ray in the rays set.
        """
        if single_point.s not in list(range(self.rays + 1)):
            return False
        return True

    def zero_check(self, single_point):
        r"""Check if a random point satisfy the zero condition.

        Parameters
        ----------
        point : SpiderPoint
             Point to be checked.

        Returns
        -------
        belongs : boolean
            Boolean denoting if the point has zero length when it has zero ray.
        """
        if single_point.s == 0 and single_point.x != 0:
            return False
        return True

    def value_check(self, single_point):
        r"""Check if a random point has the correct length.

        Parameters
        ----------
        point : SpiderPoint
             Point to be checked.

        Returns
        -------
        belongs : boolean
            Boolean denoting if the point has a positive length when on non-zero ray.
        """
        if single_point.s != 0 and single_point.x <= 0:
            return False
        return True

    def set_to_array(self, point):
        r"""Turn a point into an array compatible with the dimension of the space.

        Parameters
        ----------
        point : SpiderPoint
             Point to be checked.

        Returns
        -------
        point_array : array-like, shape=[...,rays]
            An array with the x parameter in the s position.
        """
        point_to_array = gs.array(0, self.rays)
        point_to_array[point.s] = point.x
        return point_to_array


class SpiderGeometry(PointSetGeometry, Spider):
    """Geometry on the Spider, induced by the Euclidean Geometry along the rays."""

    def __init__(self):
        super(SpiderGeometry)
        self.rays_geometry = Euclidean.metric

    def dist(self, point_a, point_b, **kwargs):
        """Compute the distance between two points."""
        if point_a.s == point_b.s or point_a.s == 0 or point_b.s == 0:
            return self.rays_geometry.dist(point_a.x, point_b.x)
        return point_a.x + point_b.x

    def geodesic(self, point_a, point_b, **kwargs):
        """Compute points on the geodesic between two points.

        Parameters
        ----------
        point_a : SpiderPoint
             Point in the Spider.
        point_b : SpiderPoint
             Point in the Spider.
        t : float between 0 and 1
            The portion of time of the returned point.
            Default 0.5.

        Returns
        -------
        point_array : array-like, shape=[...,rays]
            An array with the x parameter in the s position.
        """
        t = kwargs["t"] if "t" in kwargs else 0.5
        if point_a.s == point_b.s or point_a.s == 0 or point_b.s == 0:
            s = point_a.s if point_b.s == 0 else point_a.s
            g = self.rays_geometry.geodesic(
                initial_point=gs.array([point_a.x]), end_point=gs.array([point_b.x])
            )
            return SpiderPoint(s=s, x=float(g(t)))
        g = self.rays_geometry.geodesic(
            initial_point=gs.array([-point_a.x]), end_point=gs.array([point_b.x])
        )
        x = float(g(t))
        if x < 0:
            return SpiderPoint(s=point_a.s, x=-x)
        if x > 0:
            return SpiderPoint(s=point_b.s, x=x)
        return SpiderPoint(s=0, x=0.0)
