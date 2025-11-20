import pytest

from geomstats.geometry.scalar_product_metric import (
    ScalarProductMetric,
    _ScaledMethodsRegistry,
    _wrap_attr,
    register_scaled_method,
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
    @pytest.mark.random
    def test_scalar_metric_multiplication(self, scale):
        scaled_metric_1 = scale * self.space.metric
        scaled_metric_2 = self.space.metric * scale

        point_a, point_b = self.space.random_point(2)
        dist = self.space.metric.squared_dist(point_a, point_b)
        dist_1 = scaled_metric_1.squared_dist(point_a, point_b)
        dist_2 = scaled_metric_2.squared_dist(point_a, point_b)

        self.assertAllClose(scale * dist, dist_1)
        self.assertAllClose(scale * dist, dist_2)

        with pytest.raises(ValueError):
            scale * scaled_metric_1

    def test_scalar_metric_multiplication_error(self, scale):
        with pytest.raises(TypeError):
            scale * self.space.metric

        with pytest.raises(TypeError):
            self.space.metric * scale

        with pytest.raises(TypeError):
            self.space.metric * self.space.metric

    def test_scalar_metric_signature(self, scale):
        with pytest.raises(TypeError):
            ScalarProductMetric(self.space.metric, scale)

    @pytest.mark.random
    def test_scaling_scalar_metric(self, scale):
        point_a, point_b = self.space.random_point(2)

        dist = self.space.metric.squared_dist(point_a, point_b)

        self.space.equip_with_metric(ScalarProductMetric(self.space, scale))
        dist_1 = self.space.metric.squared_dist(point_a, point_b)

        self.space.equip_with_metric(ScalarProductMetric(self.space, scale))
        dist_2_a = self.space.metric.squared_dist(point_a, point_b)

        self.space.equip_with_metric().equip_with_metric(
            ScalarProductMetric(self.space, scale)
        ).equip_with_metric(scale * self.space.metric)
        dist_2_b = self.space.metric.squared_dist(point_a, point_b)

        self.space.equip_with_metric().equip_with_metric(
            ScalarProductMetric(self.space, scale)
        ).equip_with_metric(self.space.metric * scale)
        dist_2_c = self.space.metric.squared_dist(point_a, point_b)

        self.assertAllClose(scale * dist, dist_1)
        self.assertAllClose(scale**2 * dist, dist_2_a)
        self.assertAllClose(dist_2_b, dist_2_a)
        self.assertAllClose(dist_2_c, dist_2_a)


class CustomizationTestCase(TestCase):
    def test_register_scaled_method_invalid(self, func_name, scaling_type):
        with pytest.raises(ValueError):
            register_scaled_method(func_name, scaling_type)

    def test_register_scaled_method_valid(self, scale):
        def dummy_method(self):
            return 1

        self.Metric.dummy_method = dummy_method
        register_scaled_method("dummy_method", "linear")
        scaled_metric_1 = scale * self.space.metric
        self.assertAllClose(scale, scaled_metric_1.dummy_method())
