"""Core parametrizer classes for Stratified Spaces.
"""

import geomstats.backend as gs
import geomstats.tests

from tests.conftest import TestCase


@geomstats.tests.np_only
class PointSetTestCase(TestCase):

    def test_random_point_belongs(self, space_args, n_points):
        # TODO: add belongs_atol?
        # TODO: it is a copy of ManifoldTestCase
        space = self._PointSet(*space_args)
        random_point = space.random_point(n_points)
        result = gs.all(space.belongs(random_point))
        self.assertAllClose(result, True)

    def test_belongs(self, space_args, points, expected):
        space = self._PointSet(*space_args)
        self.assertAllClose(space.belongs(points), expected)

    def test_set_to_array(self, space_args, points, expected):
        space = self._PointSet(*space_args)

        self.assertAllClose(space.set_to_array(points), expected)


@geomstats.tests.np_only
class PointTestCase(TestCase):

    def test_to_array(self, point_args, expected):
        pt = self._Point(*point_args)
        self.assertAllClose(pt.to_array(), expected)


@geomstats.tests.np_only
class PointSetGeometryTestCase(TestCase):

    def test_dist(self, space_args, start_point, end_point, expected):
        space = self._PointSet(*space_args)
        geom = self._SetGeometry(space)
        results = geom.dist(start_point, end_point)

        self.assertAllClose(results, expected)

    def test_geodesic(self, space_args, start_point, end_point, t, expected):

        space = self._PointSet(*space_args)

        geom = self._SetGeometry(space)
        geodesic = geom.geodesic(start_point, end_point)
        pts_result = geodesic(t)

        # check results
        results = []
        for pts in pts_result:
            t_results = []
            for pt in pts:
                t_results.append(pt.to_array())

            results.append(t_results)

        results = gs.array(results)
        self.assertAllClose(results, expected)
