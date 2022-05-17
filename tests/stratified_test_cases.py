"""Core parametrizer classes for Stratified Spaces.
"""

from collections.abc import Iterable

import geomstats.backend as gs
from tests.conftest import TestCase, np_only


@np_only
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


@np_only
class PointTestCase(TestCase):
    def test_to_array(self, point_args, expected):
        pt = self.testing_data._Point(*point_args)
        self.assertAllClose(pt.to_array(), expected)


@np_only
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
        geom = self.testing_data._SetGeometry(space)
        results = geom.dist(point_a, point_b)

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
        dist_ab, dist_ba = dist_fnc([point_a], [point_b]), dist_fnc(
            [point_b], [point_a]
        )
        self.assertAllClose(dist_ab, dist_ba)

        self.assertAllClose(dist_fnc([point_a], [point_a]), gs.zeros(1))

        dist_ac, dist_cb = dist_fnc([point_a], [point_c]), dist_fnc(
            [point_c], [point_b]
        )
        rhs = dist_ac + dist_cb
        assert dist_ab <= (gs.atol + gs.rtol * rhs) + rhs

    def test_geodesic(self, space_args, start_point, end_point, t, expected):
        space = self.testing_data._PointSet(*space_args)

        geom = self.testing_data._SetGeometry(space)
        geodesic = geom.geodesic(start_point, end_point)
        pts_result = geodesic(t)

        is_list = type(start_point) is list or type(end_point) is list
        results = self._convert_to_gs_array(pts_result, is_list)
        self.assertAllClose(results, expected)

    def test_geodesic_output_shape(self, geometry, start_point, end_point, t):
        geodesic = geometry.geodesic(start_point, end_point)

        is_list = type(start_point) is list or type(end_point) is list
        n_geo = max(
            len(start_point) if type(start_point) is list else 1,
            len(end_point) if type(end_point) is list else 1,
        )
        pt = start_point[0] if start_point is list else start_point
        d_array = pt.to_array().ndim
        n_t = len(t) if type(t) is list else 1

        results = self._convert_to_gs_array(geodesic(t), is_list)
        self.assertTrue(results.ndim == d_array + 1 + int(is_list))
        self.assertTrue(results.shape[-d_array - 1] == n_t)
        if is_list:
            self.assertTrue(results.shape[-d_array - 2] == n_geo)

    def test_geodesic_bounds(self, geometry, start_point, end_point):
        geodesic = geometry.geodesic(start_point, end_point)

        results = geodesic([0.0, 1.0])
        for pt, pt_res in zip([start_point, end_point], results):
            self.assertAllClose(pt_res.to_array(), pt.to_array())
