"""Test cases for point sets."""

import geomstats.backend as gs
from geomstats.test.test_case import TestCase
from geomstats.test_cases.geometry.mixins import (
    DistTestCaseMixins,
    GeodesicBVPTestCaseMixins,
)


def _as_list(point):
    return point if isinstance(point, list) else [point]


class PointTestCase(TestCase):
    def test_point_is_equal_to_itself(self, point, atol=gs.atol):
        """Check a point is equal to itself.

        Parameters
        ----------
        point : Point or PointBatch
            Point.
        atol : float
            Absolute tolerance.
        """
        self.assertTrue(gs.all(point.equal(point, atol=atol)))


class PointSetTestCase(TestCase):
    def test_belongs(self, point, expected, atol=gs.atol):
        """Check belongs.

        Parameters
        ----------
        point : Point or PointBatch
            Point.
        expected : array-like, shape=[...]
            Expected boolean.
        atol : float
            Absolute tolerance.
        """
        res = self.space.belongs(point, atol=atol)
        self.assertAllEqual(res, expected)

    def test_random_point_belongs(self, point, atol=gs.atol):
        """Check a random point belongs to the point set.

        Parameters
        ----------
        point : Point or PointBatch
            Random point.
        atol : float
            Absolute tolerance.
        """
        self.assertTrue(gs.all(self.space.belongs(point, atol=atol)))


class PointSetMetricTestCase(DistTestCaseMixins, TestCase):
    def test_geodesic_boundary_points(self, initial_point, end_point, atol=gs.atol):
        """Check geodesic passes through its boundary points.

        Parameters
        ----------
        initial_point : Point or PointBatch
            Initial point of the geodesic.
        end_point : Point or PointBatch
            End point of the geodesic.
        atol : float
            Absolute tolerance.
        """
        geod_func = self.space.metric.geodesic(initial_point, end_point)
        geod_points = geod_func(gs.array([0.0, 1.0]))

        if not isinstance(initial_point, list):
            geod_points = [geod_points]

        for geod_point_, initial_point_, end_point_ in zip(
            geod_points, _as_list(initial_point), _as_list(end_point)
        ):
            self.assertTrue(initial_point_.equal(geod_point_[0], atol=atol))
            self.assertTrue(end_point_.equal(geod_point_[1], atol=atol))

    def test_geodesic_bvp_reverse(self, initial_point, end_point, time, atol=gs.atol):
        """Check geodesic is the reverse of the reversed geodesic.

        Parameters
        ----------
        initial_point : Point or PointBatch
            Initial point of the geodesic.
        end_point : Point or PointBatch
            End point of the geodesic.
        time : array-like, shape=[n_times,]
            Times at which to evaluate the geodesics.
        atol : float
            Absolute tolerance.
        """
        geod_func = self.space.metric.geodesic(initial_point, end_point=end_point)
        geod_func_reverse = self.space.metric.geodesic(
            end_point, end_point=initial_point
        )

        geod_point = geod_func(time)
        geod_point_reverse = geod_func_reverse(1.0 - time)

        if not isinstance(initial_point, list):
            geod_point = [geod_point]
            geod_point_reverse = [geod_point_reverse]

        for geod_point_, geod_point_reverse_ in zip(geod_point, geod_point_reverse):
            self.assertTrue(gs.all(geod_point_.equal(geod_point_reverse_, atol)))


class PointSetMetricWithArrayTestCase(
    DistTestCaseMixins, GeodesicBVPTestCaseMixins, TestCase
):
    def test_geodesic(self, initial_point, end_point, time, expected, atol=gs.atol):
        """Check geodesic.

        Parameters
        ----------
        initial_point : array-like, shape=[..., *space.shape]
            Initial point of the geodesic.
        end_point : array-like, shape=[..., *space.shape]
            End point of the geodesic.
        time : array-like, shape=[n_times,]
            Times at which to evaluate the geodesic.
        expected : array-like, shape=[..., n_times, *space.shape]
            Expected points of the geodesic.
        atol : float
            Absolute tolerance.
        """
        geod_func = self.space.metric.geodesic(initial_point, end_point=end_point)
        res = geod_func(time)
        self.assertAllClose(res, expected, atol=atol)
