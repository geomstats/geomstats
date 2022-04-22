"""Class for the spider.

Lead authors: Anna Calissano & Jonas Lueg
"""
import itertools

import geomstats.backend as gs
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.stratified.point_set import (
    Point,
    PointSet,
    PointSetMetric,
    _vectorize_point,
)


class SpiderPoint(Point):
    r"""Class for points of the Spider.

    A point in the Spider is `math:`(s,c) \in \mathbb{N} \times \mathbb{R}`.

    Parameters
    ----------
    stratum : int
        The stratum, an integer indicating the stratum the point lies in. If zero, then
        the point is on the origin.
    stratum_coord : float
        A positive number, the coordinate of the point. It must be zero if and only if
        the stratum is zero, i.e. the origin.
    """

    def __init__(self, stratum, stratum_coord):
        super(SpiderPoint, self).__init__()
        if stratum == 0 and stratum_coord != 0:
            raise ValueError("If the stratum is zero, x must be zero.")
        self.stratum = stratum
        self.stratum_coord = stratum_coord

    def __repr__(self):
        """Return a readable representation of the instance."""
        return f"s{self.stratum}: {self.stratum_coord}"

    def __hash__(self):
        """Return the hash of the instance."""
        return hash((self.stratum, self.stratum_coord))

    def __eq__(self, other):
        """Compare two points."""
        return (
            self.stratum == other.stratum
            and abs(self.stratum_coord - other.stratum_coord) < gs.atol
        )

    def to_array(self):
        """Return the hash of the instance."""
        return gs.array([self.stratum, self.stratum_coord])


class Spider(PointSet):
    r"""Spider: a set of rays attached to the origin.

    The k-spider consists of k copies of the positive real line
    `math:`\mathbb{R}_{\geq 0}` glued together at the origin [Feragen2020].

    Parameters
    ----------
    n_rays : int
        Number of rays to attach to the origin.
        Note that zero counts as the origin not as a ray.

    References
    ----------
    ..[Feragen2020]  Feragen, Aasa, and Tom Nye. "Statistics on stratified spaces."
    Riemannian Geometric Statistics in Medical Image Analysis.
    Academic Press, 2020. 299-342.
    """

    def __init__(self, n_rays):
        super(Spider, self).__init__()
        self.n_rays = n_rays

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
        if self.n_rays != 0:
            s = gs.random.randint(low=0, high=self.n_rays, size=n_samples)
            x = gs.abs(gs.random.normal(loc=10, scale=1, size=n_samples))
            x[s == 0] = 0
            return [
                SpiderPoint(stratum=s[k], stratum_coord=x[k]) for k in range(n_samples)
            ]
        return [SpiderPoint(stratum=0, stratum_coord=0)] * n_samples

    @_vectorize_point((1, "point"))
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
                self._coord_check(single_point)
                and self._n_rays_check(single_point)
                and self._zero_check(single_point)
                and type(single_point) is SpiderPoint
            ]
        return gs.array(results)

    def _n_rays_check(self, single_point):
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
        if single_point.stratum not in list(range(self.n_rays + 1)):
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
        if single_point.stratum == 0 and single_point.stratum_coord != 0:
            return False
        return True

    @staticmethod
    def _coord_check(single_point):
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
        if single_point.stratum != 0 and single_point.stratum_coord <= 0:
            return False
        return True

    @_vectorize_point((1, "point"))
    def set_to_array(self, point):
        r"""Turn a point into an array compatible with the dimension of the space.

        Parameters
        ----------
        point : SpiderPoint or list of SpiderPoint, shape=[...]
             Points to be checked.

        Returns
        -------
        point_array : array-like, shape=[...,n_rays]
            An array with the stratum_coord parameter in the stratum position.
        """
        point_to_array = gs.zeros((len(point), self.n_rays))
        for i, pt in enumerate(point):
            point_to_array[i, pt.stratum - 1] = pt.stratum_coord
        return point_to_array


class SpiderMetric(PointSetMetric):
    """Geometry on the Spider, induced by the rays Geometry."""

    def __init__(self, space, ray_metric=EuclideanMetric(1)):
        super(SpiderMetric, self).__init__(space=space)
        self.ray_metric = ray_metric

    @property
    def n_rays(self):
        """Get number of rays."""
        return self.space.n_rays

    @_vectorize_point((1, "a"), (2, "b"))
    def dist(self, point_a, point_b):
        """Compute the distance between two points on the Spider using the ray geometry.

        The spider metric is the metric in each ray extended to the Spider:
        given two points x, y on different rays, d(x, y) = d(x, 0) + d(0, y).

        Parameters
        ----------
        point_a : SpiderPoint or list of SpiderPoint, shape=[...]
             Point in the Spider.
        point_b : SpiderPoint or list of SpiderPoint, shape=[...]
             Point in the Spider.

        Returns
        -------
        point_array : array-like, shape=[...]
            An array with the distance.
        """
        if len(point_a) == 1:
            values = itertools.zip_longest(point_a, point_b, fillvalue=point_a[0])
        elif len(point_b) == 1:
            values = itertools.zip_longest(point_a, point_b, fillvalue=point_b[0])
        else:
            values = itertools.zip_longest(point_a, point_b)

        result = []
        for point_a_, point_b_ in values:
            if (
                point_a_.stratum == point_b_.stratum
                or point_a_.stratum == 0
                or point_b_.stratum == 0
            ):
                result += [
                    self.ray_metric.norm(
                        gs.array([point_a_.stratum_coord - point_b_.stratum_coord])
                    )
                ]
            else:
                result += [point_a_.stratum_coord + point_b_.stratum_coord]
        return gs.array(result) if len(result) != 1 else result[0]

    @_vectorize_point((1, "initial_point"), (2, "end_point"))
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

    def _point_geodesic(self, initial_point, end_point):
        """Compute the distance between two Spider points.

        Parameters
        ----------
        initial_point : SpiderPoint
             Point in the Spider.
        end_point : SpiderPoint
             Point in the Spider.

        Returns
        -------
        geo: function
            Geodesic between two Spider Points.
        """
        if (
            initial_point.stratum == end_point.stratum
            or initial_point.stratum == 0
            or end_point.stratum == 0
        ):
            s = gs.maximum(initial_point.stratum, end_point.stratum)

            def ray_geo(t):
                g = self.ray_metric.geodesic(
                    initial_point=gs.array([initial_point.stratum_coord]),
                    end_point=gs.array([end_point.stratum_coord]),
                )

                x = g(t)
                return [
                    SpiderPoint(stratum=s if xx[0] else 0, stratum_coord=xx[0])
                    for xx in x
                ]

            return ray_geo

        def ray_geo(t):
            g = self.ray_metric.geodesic(
                initial_point=gs.array([-initial_point.stratum_coord]),
                end_point=gs.array([end_point.stratum_coord]),
            )
            x = g(t)
            return [
                SpiderPoint(stratum=initial_point.stratum, stratum_coord=-xx[0])
                if xx < 0.0
                else SpiderPoint(stratum=end_point.stratum, stratum_coord=xx[0])
                for xx in x
            ]

        return ray_geo
