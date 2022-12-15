import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.scalar_product_metric import (
    ScalarProductMetric,
    _get_scaling_factor,
    _wrap_attr,
)
from tests.conftest import TestCase


class TestWrapper(TestCase):
    def test_wrap_attr(self):
        def example_fnc():
            return 2.0

        scaled_func = _wrap_attr(3.0, example_fnc)

        res = example_fnc()
        scaled_res = scaled_func()

        self.assertAllClose(res, scaled_res / 3.0)

    def test_get_scaling_factor(self):
        scale = 2.0

        func_name = "dist"
        scaling_factor = _get_scaling_factor(func_name, scale)
        self.assertAllClose(gs.sqrt(scale), scaling_factor)

        func_name = "metric_matrix"
        scaling_factor = _get_scaling_factor(func_name, scale)
        self.assertAllClose(scale, scaling_factor)

        func_name = "cometric_matrix"
        scaling_factor = _get_scaling_factor(func_name, scale)
        self.assertAllClose(1.0 / scale, scaling_factor)

        func_name = "normalize"
        scaling_factor = _get_scaling_factor(func_name, scale)
        self.assertAllClose(1.0 / gs.power(scale, 2), scaling_factor)

        func_name = "not_scaled"
        scaling_factor = _get_scaling_factor(func_name, scale)
        assert scaling_factor is None


class TestInstantiation(TestCase):
    def test_scalar_metric_multiplication(self):
        space = Euclidean(dim=3)
        scale = 2.0

        scaled_metric_1 = scale * space.metric
        scaled_metric_2 = space.metric * scale

        point_a, point_b = space.random_point(2)
        dist = space.metric.squared_dist(point_a, point_b)
        dist_1 = scaled_metric_1.squared_dist(point_a, point_b)
        dist_2 = scaled_metric_2.squared_dist(point_a, point_b)

        self.assertAllClose(scale * dist, dist_1)
        self.assertAllClose(scale * dist, dist_2)

    def test_scaling_scalar_metric(self):
        space = Euclidean(dim=3)
        scale = 2.0

        scaled_metric_1 = ScalarProductMetric(space.metric, scale)
        scaled_metric_2_a = ScalarProductMetric(scaled_metric_1, scale)
        scaled_metric_2_b = scale * scaled_metric_1
        scaled_metric_2_c = scaled_metric_1 * scale

        point_a, point_b = space.random_point(2)
        dist = space.metric.squared_dist(point_a, point_b)
        dist_1 = scaled_metric_1.squared_dist(point_a, point_b)
        dist_2_a = scaled_metric_2_a.squared_dist(point_a, point_b)
        dist_2_b = scaled_metric_2_b.squared_dist(point_a, point_b)
        dist_2_c = scaled_metric_2_c.squared_dist(point_a, point_b)

        self.assertAllClose(scale * dist, dist_1)
        self.assertAllClose(scale**2 * dist, dist_2_a)
        self.assertAllClose(dist_2_b, dist_2_a)
        self.assertAllClose(dist_2_c, dist_2_a)

    def test_assigning_scalar_metric_to_space(self):
        space = Euclidean(dim=3)
        scale = 2.0
        scaled_metric = ScalarProductMetric(space.metric, scale)
        space.metric = scaled_metric
