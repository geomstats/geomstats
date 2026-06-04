"""Core parametrizer classes for Stratified Spaces."""

import pytest

import geomstats.backend as gs
from geomstats.test.random import get_random_times
from geomstats.test.test_case import TestCase
from geomstats.test_cases.geometry.mixins import (
    DistTestCaseMixins,
    GeodesicBVPTestCaseMixins,
)


class RandomDataGenerator:
    def __init__(self, space):
        self.space = space

    def random_point(self, n_points=1):
        return self.space.random_point(n_points)


class PointTestCase(TestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)

    @pytest.mark.random
    def test_point_is_equal_to_itself(self, n_points, atol):
        initial_point = self.data_generator.random_point(n_points)

        self.assertTrue(gs.all(initial_point.equal(initial_point, atol=atol)))


class PointSetTestCase(TestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)

    def test_belongs(self, point, expected, atol):
        res = self.space.belongs(point, atol=atol)
        self.assertAllEqual(res, expected)

    @pytest.mark.random
    def test_random_point_belongs(self, n_points, atol):
        random_point = self.space.random_point(n_points)

        expected = gs.ones(n_points)
        self.test_belongs(random_point, expected, atol=atol)


class PointSetMetricTestCase(DistTestCaseMixins, TestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)

    @pytest.mark.random
    def test_geodesic_boundary_points(self, n_points, atol):
        initial_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)

        time = gs.array([0.0, 1.0])

        geod_func = self.space.metric.geodesic(initial_point, end_point)

        geod_points = geod_func(time)

        if n_points == 1:
            initial_point = [initial_point]
            end_point = [end_point]
            geod_points = [geod_points]

        for geod_point_, initial_point_, end_point_ in zip(
            geod_points, initial_point, end_point
        ):
            self.assertTrue(initial_point_.equal(geod_point_[0], atol=atol))
            self.assertTrue(end_point_.equal(geod_point_[1], atol=atol))

    @pytest.mark.random
    def test_geodesic_bvp_reverse(self, n_points, n_times, atol):
        initial_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)

        time = get_random_times(n_times)

        geod_func = self.space.metric.geodesic(initial_point, end_point=end_point)
        geod_func_reverse = self.space.metric.geodesic(
            end_point, end_point=initial_point
        )

        geod_point = geod_func(time)
        geod_point_reverse = geod_func_reverse(1.0 - time)

        if n_points == 1:
            geod_point = [geod_point]
            geod_point_reverse = [geod_point_reverse]

        for geod_point_, geod_point_reverse_ in zip(geod_point, geod_point_reverse):
            self.assertTrue(gs.all(geod_point_.equal(geod_point_reverse_, atol)))


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
