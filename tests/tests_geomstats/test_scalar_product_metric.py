import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.scalar_product_metric import (
    ScalarProductMetric,
    _get_scaling_factor,
    _wrap_attr,
)
from tests.conftest import TestCase


class ScalarProductMetricTestCase(TestCase):
    def test_dist(self):
        point_a, point_b = self.space.random_point(2)

        dist = self.space.metric.dist(point_a, point_b)
        scaled_dist = self.scaled_metric.dist(point_a, point_b)

        self.assertAllClose(
            dist, scaled_dist / gs.sqrt(self.scaled_metric.scaling_factor)
        )


class TestScalarProductMetricEuclidean(ScalarProductMetricTestCase):
    space = Euclidean(dim=3)
    scaled_metric = ScalarProductMetric(space.metric, 2.0)


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

        scaled_metric_1 = 2.0 * space.metric
        scaled_metric_2 = space.metric * 2.0

        point_a, point_b = space.random_point(2)
        dist = space.metric.squared_dist(point_a, point_b)
        dist_1 = scaled_metric_1.squared_dist(point_a, point_b)
        dist_2 = scaled_metric_2.squared_dist(point_a, point_b)

        self.assertAllClose(2.0 * dist, dist_1)
        self.assertAllClose(2.0 * dist, dist_2)
