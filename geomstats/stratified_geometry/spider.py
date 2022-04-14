"""Class for the spider.

Lead authors: Anna Calissano & Jonas Lueg
"""
import itertools

import geomstats.backend as gs
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.stratified_geometry.stratified_spaces import (
    Point,
    PointSet,
    PointSetGeometry,
    _vectorize_point,
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
        super(SpiderPoint, self).__init__()
        if s == 0 and x != 0:
            raise ValueError("If the stratum is zero, x must be zero.")
        self.s = s
        self.x = x

    def __repr__(self):
        """Return a readable representation of the instance."""
        return f"s{self.s}: {self.x}"

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
        samples : list of SpiderPoint, shape=[...]
            List of SpiderPoints randomly sampled from the Spider.
        """
        if self.rays != 0:
            s = gs.random.randint(low=0, high=self.rays, size=n_samples)
            x = gs.abs(gs.random.normal(loc=10, scale=1, size=n_samples))
            x[s == 0] = 0
            return [SpiderPoint(s=s[k], x=x[k]) for k in range(n_samples)]
        return [SpiderPoint(s=0, x=0)] * n_samples

    @_vectorize_point((1, 'point'))
    def belongs(self, point):
        r"""Check if a random point belongs to the spider set.

        Parameters
        ----------
        point : SpiderPoint or list of SpiderPoint, shape=[...]
             Point to be checked.

        Returns
        -------
        belongs : array-like, shape=[...]
            Boolean denoting if the SpiderPoint belongs to the Spider Set.
        """
        results = []
        for single_point in point:
            results += [
                self._value_check(single_point)
                and self._rays_check(single_point)
                and self._zero_check(single_point)
                and type(single_point) is SpiderPoint
            ]
        return gs.array(results)

    def _rays_check(self, single_point):
        r"""Check if a random point has the correct number of rays.

        Parameters
        ----------
        single_point : SpiderPoint
             Point to be checked.

        Returns
        -------
        belongs : boolean
            Boolean denoting if the point has a ray in the rays set.
        """
        if single_point.s not in list(range(self.rays + 1)):
            return False
        return True

    @staticmethod
    def _zero_check(single_point):
        r"""Check if a random point satisfy the zero condition.

        Parameters
        ----------
        single_point : SpiderPoint
             Point to be checked.

        Returns
        -------
        belongs : boolean
            Boolean denoting if the point has zero length when it has zero ray.
        """
        if single_point.s == 0 and single_point.x != 0:
            return False
        return True

    @staticmethod
    def _value_check(single_point):
        r"""Check if a random point has the correct length.

        Parameters
        ----------
        single_point : SpiderPoint
             Point to be checked.

        Returns
        -------
        belongs : boolean
            Boolean denoting if the point has a positive length when on non-zero ray.
        """
        if single_point.s != 0 and single_point.x <= 0:
            return False
        return True

    @_vectorize_point((1, 'point'))
    def set_to_array(self, point):
        r"""Turn a point into an array compatible with the dimension of the space.

        Parameters
        ----------
        point : SpiderPoint or list of SpiderPoint, shape=[...]
             Points to be checked.

        Returns
        -------
        point_array : array-like, shape=[...,rays]
            An array with the x parameter in the s position.
        """
        point_to_array = gs.zeros((len(point), self.rays))
        for i, pt in enumerate(point):
            point_to_array[i, pt.s - 1] = pt.x
        return point_to_array


class SpiderGeometry(PointSetGeometry):
    """Geometry on the Spider, induced by the rays Geometry."""

    def __init__(self, space, ambient_metric=EuclideanMetric(1)):
        super(SpiderGeometry, self).__init__(space=space)
        self.rays_geometry = ambient_metric

    @property
    def rays(self):
        return self.space.rays

    @_vectorize_point((1, 'a'), (2, 'b'))
    def dist(self, a, b):
        """Compute the distance between two points on the Spider using the ray geometry.

        Parameters
        ----------
        a : SpiderPoint or list of SpiderPoint, shape=[...]
             Point in the Spider.
        b : SpiderPoint or list of SpiderPoint, shape=[...]
             Point in the Spider.

        Returns
        -------
        point_array : array-like, shape=[...]
            An array with the distance.
        """
        result = []
        if len(a) == 1:
            values = itertools.zip_longest(a, b, fillvalue=a[0])
        elif len(b) == 1:
            values = itertools.zip_longest(a, b, fillvalue=b[0])
        else:
            values = itertools.zip_longest(a, b)
        for point_a, point_b in values:
            if point_a.s == point_b.s or point_a.s == 0 or point_b.s == 0:
                result += [self.rays_geometry.norm(gs.array([point_a.x - point_b.x]))]
            else:
                result += [point_a.x + point_b.x]
        return gs.array(result)

    @_vectorize_point((1, 'initial_point'), (2, 'end_point'))
    def geodesic(self, initial_point, end_point):
        """Return the geodesic between two lists of Spider points.

        Parameters
        ----------
        initial_point : SpiderPoint or list of SpiderPoint, shape=[...]
             Point in the Spider.
        end_point : SpiderPoint or list of SpiderPoint, shape=[...]
             Point in the Spider.

        Returns
        -------
        geo : function
            Return a vectorized geodesic function.
        """

        def _vec(t, fncs):
            if len(fncs) == 1:
                return fncs[0](t)

            return [fnc(t) for fnc in fncs]

        if len(initial_point) == 1 and len(end_point) != 1:
            values = itertools.zip_longest(
                initial_point, end_point, fillvalue=initial_point[0]
            )
        else:
            values = zip(initial_point, end_point)
        fncs = [self._point_geodesic(pt_a, pt_b) for (pt_a, pt_b) in values]
        return lambda t: _vec(t, fncs=fncs)

    def _point_geodesic(self, point_a, point_b):
        """Compute the distance between two Spider points.

        Parameters
        ----------
        point_a : SpiderPoint
             Point in the Spider.
        point_b : SpiderPoint
             Point in the Spider.

        Returns
        -------
        geo: function
            Geodesic between two Spider Points.
        """
        if point_a.s == point_b.s or point_a.s == 0 or point_b.s == 0:
            s = gs.maximum(point_a.s, point_b.s)

            def ray_geo(t):
                g = self.rays_geometry.geodesic(
                    initial_point=gs.array([point_a.x]),
                    end_point=gs.array([point_b.x]),
                )

                x = g(t)
                return [SpiderPoint(s=s if xx[0] else 0, x=xx[0]) for xx in x]

            return ray_geo

        def ray_geo(t):
            g = self.rays_geometry.geodesic(
                initial_point=gs.array([-point_a.x]), end_point=gs.array([point_b.x])
            )
            x = g(t)
            return [SpiderPoint(s=point_a.s, x=-xx[0]) if xx < 0.
                    else SpiderPoint(s=point_b.s, x=xx[0])
                    for xx in x]

        return ray_geo
