from geomstats.geometry.scalar_product_metric import (
    ScalarProductMetric,
    _ScaledMethodsRegistry,
    _wrap_attr,
)
from geomstats.test.test_case import TestCase


class WrapperTestCase(TestCase):
    def test_wrap_attr(self, func, scale):
        scaled_func = _wrap_attr(scale, func)

        res = func()
        scaled_res = scaled_func()

        self.assertAllClose(res, scaled_res / scale)

    def test_scaling_factor(self, func_name, scale, expected):
        scaling_factor = _ScaledMethodsRegistry._get_scaling_factor(func_name, scale)
        self.assertAllClose(scaling_factor, expected)

    def test_non_scaled(self, func_name, scale):
        scaling_factor = _ScaledMethodsRegistry._get_scaling_factor(func_name, scale)
        assert scaling_factor is None


class InstantiationTestCase(TestCase):
    def test_scalar_metric_multiplication(self, scale):
        scaled_metric_1 = scale * self.space.metric
        scaled_metric_2 = self.space.metric * scale

        point_a, point_b = self.space.random_point(2)
        dist = self.space.metric.squared_dist(point_a, point_b)
        dist_1 = scaled_metric_1.squared_dist(point_a, point_b)
        dist_2 = scaled_metric_2.squared_dist(point_a, point_b)

        self.assertAllClose(scale * dist, dist_1)
        self.assertAllClose(scale * dist, dist_2)

    def test_scaling_scalar_metric(self, scale):
        scaled_metric_1 = ScalarProductMetric(self.space.metric, scale)
        scaled_metric_2_a = ScalarProductMetric(scaled_metric_1, scale)
        scaled_metric_2_b = scale * scaled_metric_1
        scaled_metric_2_c = scaled_metric_1 * scale

        point_a, point_b = self.space.random_point(2)
        dist = self.space.metric.squared_dist(point_a, point_b)
        dist_1 = scaled_metric_1.squared_dist(point_a, point_b)
        dist_2_a = scaled_metric_2_a.squared_dist(point_a, point_b)
        dist_2_b = scaled_metric_2_b.squared_dist(point_a, point_b)
        dist_2_c = scaled_metric_2_c.squared_dist(point_a, point_b)

        self.assertAllClose(scale * dist, dist_1)
        self.assertAllClose(scale**2 * dist, dist_2_a)
        self.assertAllClose(dist_2_b, dist_2_a)
        self.assertAllClose(dist_2_c, dist_2_a)
