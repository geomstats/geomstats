import pytest

import geomstats.backend as gs
from geomstats.test.random import RandomDataGenerator
from geomstats.test.test_case import TestCase
from geomstats.test_cases.geometry.stratified.point_set import PointTestCase


class RandomGroveDataGenerator(RandomDataGenerator):
    def __init__(self, space, topology):
        super().__init__(space)
        self.topology = topology

    def random_point(self, n_points=1):
        return self.space.random_grove_point(self.topology, n_points)


class WaldTestCase(PointTestCase):
    def test_corr(self, point, expected, atol):
        self.assertAllClose(point.corr, expected, atol=atol)


class WaldGeodesicSolverTestCase(TestCase):
    @pytest.mark.random
    def test_discrete_geodesic_reverse(self, n_points, atol):
        initial_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)

        geod_points = self.geodesic_solver.discrete_geodesic(initial_point, end_point)
        geod_points_reversed = self.geodesic_solver.discrete_geodesic(
            end_point, initial_point
        )

        if n_points == 1:
            geod_points = [geod_points]
            geod_points_reversed = [geod_points_reversed]

        for geod_points_, geod_points_reversed_ in zip(
            geod_points, geod_points_reversed
        ):
            geod_points_reversed_.reverse()
            self.assertTrue(
                gs.all(geod_points_.equal(geod_points_reversed_, atol=atol))
            )
