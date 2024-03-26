"""Core parametrizer classes for Stratified Spaces."""

from collections.abc import Iterable

import geomstats.backend as gs
from geomstats.test.random import RandomDataGenerator
from geomstats.test.test_case import TestCase
from geomstats.test_cases.geometry.mixins import (
    DistTestCaseMixins,
    GeodesicBVPTestCaseMixins,
)


class PointSetTestCase(TestCase):
    def test_random_point_belongs(self, space_args, n_points):
        space = self.testing_data._PointSet(*space_args)
        random_point = space.random_point(n_points)
        result = gs.all(space.belongs(random_point))
        self.assertAllClose(result, True)

    def test_random_point_output_shape(self, space, n_samples):
        points = space.random_point(n_samples)
        self.assertTrue(len(points) == n_samples)

    def test_belongs(self, space_args, points, expected):
        space = self.testing_data._PointSet(*space_args)
        self.assertAllClose(space.belongs(points), expected)

    def test_set_to_array(self, space_args, points, expected):
        space = self.testing_data._PointSet(*space_args)
        self.assertAllClose(space.set_to_array(points), expected)

    def test_set_to_array_output_shape(self, space, points):
        n = len(points) if type(points) is list else 1
        self.assertTrue(space.set_to_array(points).shape[0] == n)


class PointTestCase(TestCase):
    def test_to_array(self, point_args, expected):
        pt = self.testing_data._Point(*point_args)
        self.assertAllClose(pt.to_array(), expected)


class PointSetMetricTestCase(TestCase):
    @staticmethod
    def _convert_to_gs_array(results, is_list):
        if is_list:
            resh_res = [[pt.to_array() for pt in pts_geo] for pts_geo in results]
        else:
            resh_res = [pt.to_array() for pt in results]

        return gs.array(resh_res)

    def test_dist(self, space_args, point_a, point_b, expected):
        space = self.testing_data._PointSet(*space_args)
        results = space.metric.dist(point_a, point_b)

        self.assertAllClose(results, expected)

    def test_dist_output_shape(self, dist_fnc, point_a, point_b):
        results = dist_fnc(point_a, point_b)

        is_array = type(point_a) is list or type(point_b) is list
        if is_array:
            n_dist = max(
                len(point_a) if type(point_a) is list else 1,
                len(point_b) if type(point_b) is list else 1,
            )
            self.assertTrue(results.size == n_dist)
        else:
            self.assertTrue(not isinstance(results, Iterable))

    def test_dist_properties(self, dist_fnc, point_a, point_b, point_c):
        dist_ab = dist_fnc(point_a, point_b)
        dist_ba = dist_fnc(point_b, point_a)
        self.assertAllClose(dist_ab, dist_ba)

        res = dist_fnc(point_a, point_a)
        self.assertAllClose(res, gs.zeros_like(res))

        dist_ac = dist_fnc(point_a, point_c)
        dist_cb = dist_fnc(point_c, point_b)
        rhs = dist_ac + dist_cb
        assert dist_ab <= (gs.atol + gs.rtol * rhs) + rhs

    def test_geodesic(self, space_args, start_point, end_point, t, expected):
        space = self.testing_data._PointSet(*space_args)

        geom = self.testing_data._PointSetMetric(space)
        geodesic = geom.geodesic(start_point, end_point)
        pts_result = geodesic(t)

        is_list = type(start_point) is list or type(end_point) is list
        results = self._convert_to_gs_array(pts_result, is_list)
        self.assertAllClose(results, expected)

    def test_geodesic_output_shape(self, metric, start_point, end_point, t):
        geodesic = metric.geodesic(start_point, end_point)

        is_list = type(start_point) is list or type(end_point) is list
        n_geo = max(
            len(start_point) if type(start_point) is list else 1,
            len(end_point) if type(end_point) is list else 1,
        )
        pt = start_point[0] if type(start_point) is list else start_point
        d_array = gs.ndim(pt.to_array())
        n_t = len(t) if type(t) is list else 1

        results = self._convert_to_gs_array(geodesic(t), is_list)
        self.assertTrue(results.ndim == d_array + 1 + int(is_list))
        self.assertTrue(results.shape[-d_array - 1] == n_t)
        if is_list:
            self.assertTrue(results.shape[-d_array - 2] == n_geo)

    def test_geodesic_bounds(self, space, start_point, end_point):
        geodesic = space.metric.geodesic(start_point, end_point)

        results = geodesic([0.0, 1.0])
        for pt, pt_res in zip([start_point, end_point], results):
            self.assertAllClose(pt_res.to_array(), pt.to_array())


class PointSetMetricWithArrayTestCase(
    DistTestCaseMixins, GeodesicBVPTestCaseMixins, TestCase
):
    tangent_to_multiple = False
    is_metric = True

    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)

    def test_geodesic(
        self,
        initial_point,
        end_point,
        time,
        expected,
        atol,
    ):
        geod_func = self.space.metric.geodesic(initial_point, end_point=end_point)
        res = geod_func(time)
        self.assertAllClose(res, expected, atol=atol)
