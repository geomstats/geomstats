"""Class for the spider.

Lead authors: Anna Calissano & Jonas Lueg
"""

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.stratified.point_set import (
    Point,
    PointBatch,
    PointSet,
    PointSetMetric,
)
from geomstats.geometry.stratified.vectorization import broadcast_lists, vectorize_point


class SpiderPoint(Point):
    r"""Class for points of the Spider.

    A point in the Spider is :math:`(s,c) \in \mathbb{N} \times \mathbb{R}`.

    Parameters
    ----------
    stratum : int
        The stratum, an integer indicating the stratum the point lies in.
        If zero, then the point is on the origin.
    coord : array-like, shape=[1,]
        A positive number, the coordinate of the point. It must be zero if and
        only if the stratum is zero, i.e. the origin.
    """

    def __init__(self, stratum, coord):
        super().__init__()
        self.stratum = stratum
        self.coord = coord

    def __repr__(self):
        """Return a readable representation of the instance."""
        return f"r{self.stratum}: {self.coord[0]}"

    def _equal_single(self, point, atol=gs.atol):
        """Check equality against another point.

        Parameters
        ----------
        point : Point
            Point to compare against.
        atol : float

        Returns
        -------
        is_equal : bool
        """
        return self.stratum == point.stratum and abs(self.coord - point.coord) < gs.atol

    @vectorize_point((1, "point"))
    def equal(self, point, atol=gs.atol):
        """Check equality against another point.

        Parameters
        ----------
        point : Point or PointBatch
            Point to compare against.
        atol : float

        Returns
        -------
        is_equal : array-like, shape=[...]
        """
        return gs.array([self._equal_single(point_, atol) for point_ in point])


class Spider(PointSet):
    r"""Spider: a set of rays attached to the origin.

    The k-spider consists of k copies of the positive real line
    :math:`\mathbb{R}_{\geq 0}` glued together at the origin [Feragen2020]_.

    Parameters
    ----------
    n_rays : int
        Number of rays to attach to the origin.

    References
    ----------
    .. [Feragen2020]  Feragen, Aasa, and Tom Nye. "Statistics on stratified spaces."
        Riemannian Geometric Statistics in Medical Image Analysis.
        Academic Press, 2020. 299-342.
    """

    def __init__(self, n_rays, equip=True):
        super().__init__(equip=equip)
        self.n_rays = n_rays
        self.stratum_space = Euclidean(dim=1)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return SpiderMetric

    def random_point(self, n_samples=1):
        r"""Compute a random point of the spider set.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        samples : SpiderPoint or PointBatch
            List of SpiderPoints randomly sampled from the Spider.
        """
        s = gs.random.randint(low=0, high=self.n_rays, size=(n_samples,))
        x = gs.abs(gs.random.normal(loc=10, scale=1, size=n_samples))
        random_point = [
            SpiderPoint(stratum=s[k], coord=gs.array([x[k]])) for k in range(n_samples)
        ]
        if n_samples == 1:
            return random_point[0]

        return PointBatch(random_point)

    @vectorize_point((1, "point"))
    def belongs(self, point, atol=gs.atol):
        r"""Check if a random point belongs to the spider set.

        Parameters
        ----------
        point : SpiderPoint or PointBatch
             Point to be checked.

        Returns
        -------
        belongs : array-like, shape=[...]
            Boolean evaluating if point belongs to the set.
        """
        results = []
        for point_ in point:
            results.append(
                self._coord_check(point_, atol) and self._n_rays_check(point_)
            )
        return gs.array(results)

    def _n_rays_check(self, point):
        r"""Check if a random point has the correct number of rays.

        Parameters
        ----------
        point : SpiderPoint
             Point to be checked.

        Returns
        -------
        belongs : bool
            Boolean denoting if the point has a ray in the rays set.
        """
        if point.stratum < self.n_rays:
            return True
        return False

    @staticmethod
    def _coord_check(point, atol=gs.atol):
        r"""Check if a random point has the correct length.

        Parameters
        ----------
        point : SpiderPoint
             Point to be checked.
        atol : float
            Absolute tolerance.

        Returns
        -------
        belongs : boolean
            Boolean denoting if the point has a positive length when on non-zero ray.
        """
        if point.coord <= -atol:
            return False
        return True


class SpiderMetric(PointSetMetric):
    """Geometry on the Spider, induced by the rays metric."""

    @property
    def _stratum_metric(self):
        return self._space.stratum_space.metric

    def _dist_single(self, point_a, point_b):
        """Compute the distance between two points on the Spider using the ray geometry.

        The spider metric is the metric in each ray extended to the Spider:
        given two points x, y on different rays, d(x, y) = d(x, 0) + d(0, y).

        Parameters
        ----------
        point_a : SpiderPoint
             Point in the Spider.
        point_b : SpiderPoint
             Point in the Spider.

        Returns
        -------
        dist : array-like, shape=[...]
            Distance between points.
        """
        if point_a.stratum == point_b.stratum:
            return self._stratum_metric.dist(point_a.coord, point_b.coord)

        return self._stratum_metric.dist(-point_a.coord, point_b.coord)

    @vectorize_point((1, "point_a"), (2, "point_b"))
    def dist(self, point_a, point_b):
        """Compute the distance between two points on the Spider using the ray geometry.

        The spider metric is the metric in each ray extended to the Spider:
        given two points x, y on different rays, d(x, y) = d(x, 0) + d(0, y).

        Parameters
        ----------
        point_a : SpiderPoint or PointBatch
             Point in the Spider.
        point_b : SpiderPoint or PointBatch
             Point in the Spider.

        Returns
        -------
        dist : array-like, shape=[...]
            Distance between points.
        """
        point_a, point_b = broadcast_lists(point_a, point_b)
        return gs.array(
            [
                self._dist_single(point_a_, point_b_)
                for point_a_, point_b_ in zip(point_a, point_b)
            ]
        )

    @vectorize_point((1, "initial_point"), (2, "end_point"))
    def geodesic(self, initial_point, end_point):
        """Return the geodesic between two lists of Spider points.

        Parameters
        ----------
        initial_point : SpiderPoint or PointBatch
             Point in the Spider.
        end_point : SpiderPoint or PointBatch
             Point in the Spider.

        Returns
        -------
        path : callable
            Return a vectorized geodesic function.
        """
        initial_point, end_point = broadcast_lists(initial_point, end_point)

        def _vec(t, fncs):
            if len(fncs) == 1:
                return fncs[0](t)

            return [fnc(t) for fnc in fncs]

        fncs = [
            self._geodesic_single(initial_point_, end_point_)
            for (initial_point_, end_point_) in zip(initial_point, end_point)
        ]
        return lambda t: _vec(t, fncs=fncs)

    def _geodesic_single(self, initial_point, end_point):
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
        if initial_point.stratum == end_point.stratum:

            def ray_geo(t):
                ray_geod_func = self._stratum_metric.geodesic(
                    initial_point=initial_point.coord,
                    end_point=end_point.coord,
                )

                ray_geod_points = ray_geod_func(t)
                return PointBatch(
                    [
                        SpiderPoint(stratum=initial_point.stratum, coord=coord)
                        for coord in ray_geod_points
                    ]
                )

            return ray_geo

        def ray_geo(t):
            pseudo_ray_geod_func = self._stratum_metric.geodesic(
                initial_point=-initial_point.coord,
                end_point=end_point.coord,
            )
            pseudo_ray_geod_points = pseudo_ray_geod_func(t)
            return PointBatch(
                [
                    (
                        SpiderPoint(stratum=initial_point.stratum, coord=-coord)
                        if coord < 0.0
                        else SpiderPoint(stratum=end_point.stratum, coord=coord)
                    )
                    for coord in pseudo_ray_geod_points
                ]
            )

        return ray_geo
