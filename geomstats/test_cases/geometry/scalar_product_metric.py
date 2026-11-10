import copy

import pytest

import geomstats.backend as gs
from geomstats.geometry.scalar_product_metric import (
    ScalarProductMetric,
    register_scaled_method,
)
from geomstats.test.test_case import TestCase


class ScalarProductMetricComparisonTestCase(TestCase):
    """Test case comparing a scaled metric with its underlying metric.

    `other_space` is `space` equipped with its metric scaled by `scale`.
    """

    def _assert_scaled(self, func_name, factor, *args, atol=gs.atol):
        res = getattr(self.other_space.metric, func_name)(*args)
        expected = factor * getattr(self.space.metric, func_name)(*args)
        self.assertAllClose(res, expected, atol=atol)

    def test_metric_matrix(self, base_point, atol=gs.atol):
        """Check metric matrix is scaled by `scale`.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        self._assert_scaled("metric_matrix", self.scale, base_point, atol=atol)

    def test_cometric_matrix(self, base_point, atol=gs.atol):
        """Check cometric matrix is scaled by `1 / scale`.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        self._assert_scaled("cometric_matrix", 1.0 / self.scale, base_point, atol=atol)

    def test_inner_product(
        self, tangent_vec_a, tangent_vec_b, base_point, atol=gs.atol
    ):
        """Check inner product is scaled by `scale`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        self._assert_scaled(
            "inner_product",
            self.scale,
            tangent_vec_a,
            tangent_vec_b,
            base_point,
            atol=atol,
        )

    def test_squared_norm(self, vector, base_point, atol=gs.atol):
        """Check squared norm is scaled by `scale`.

        Parameters
        ----------
        vector : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        self._assert_scaled("squared_norm", self.scale, vector, base_point, atol=atol)

    def test_norm(self, vector, base_point, atol=gs.atol):
        """Check norm is scaled by `sqrt(scale)`.

        Parameters
        ----------
        vector : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        self._assert_scaled("norm", gs.sqrt(self.scale), vector, base_point, atol=atol)

    def test_normalize(self, vector, base_point, atol=gs.atol):
        """Check normalized vector is scaled by `1 / sqrt(scale)`.

        Parameters
        ----------
        vector : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        self._assert_scaled(
            "normalize", 1.0 / gs.sqrt(self.scale), vector, base_point, atol=atol
        )

    def test_squared_dist(self, point_a, point_b, atol=gs.atol):
        """Check squared distance is scaled by `scale`.

        Parameters
        ----------
        point_a : array-like, shape=[..., *space.shape]
            Point on the manifold.
        point_b : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        self._assert_scaled("squared_dist", self.scale, point_a, point_b, atol=atol)

    def test_dist(self, point_a, point_b, atol=gs.atol):
        """Check distance is scaled by `sqrt(scale)`.

        Parameters
        ----------
        point_a : array-like, shape=[..., *space.shape]
            Point on the manifold.
        point_b : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        self._assert_scaled("dist", gs.sqrt(self.scale), point_a, point_b, atol=atol)

    def test_injectivity_radius(self, base_point, atol=gs.atol):
        """Check injectivity radius is scaled by `sqrt(scale)`.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        self._assert_scaled(
            "injectivity_radius", gs.sqrt(self.scale), base_point, atol=atol
        )

    def test_exp(self, tangent_vec, base_point, atol=gs.atol):
        """Check exponential is not scaled.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        self._assert_scaled("exp", 1.0, tangent_vec, base_point, atol=atol)

    def test_log(self, point, base_point, atol=gs.atol):
        """Check logarithm is not scaled.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        self._assert_scaled("log", 1.0, point, base_point, atol=atol)


class MetricScalingTestCase(TestCase):
    def test_scalar_metric_multiplication(self, scale):
        """Check ``scale * metric`` and ``metric * scale`` wrap the metric.

        Parameters
        ----------
        scale : float
            Scale of the metric.
        """
        for scaled_metric in (scale * self.space.metric, self.space.metric * scale):
            assert isinstance(scaled_metric, ScalarProductMetric)
            self.assertAllClose(scaled_metric.scale, scale)
            assert scaled_metric.underlying_metric is self.space.metric

    def test_scalar_metric_multiplication_not_equipped(self, scale):
        """Check a scaled metric cannot be scaled before equipping the space.

        Parameters
        ----------
        scale : float
            Scale of the metric.
        """
        with pytest.raises(ValueError):
            scale * (scale * self.space.metric)

    def test_scalar_metric_multiplication_error(self, scale):
        """Check multiplying a metric by a non-float raises.

        Parameters
        ----------
        scale : any
            Value that is not a float.
        """
        with pytest.raises(TypeError):
            scale * self.space.metric

        with pytest.raises(TypeError):
            self.space.metric * scale

        with pytest.raises(TypeError):
            self.space.metric * self.space.metric

    def test_scalar_metric_signature(self, scale):
        """Check instantiating with a metric instead of a space raises.

        Parameters
        ----------
        scale : float
            Scale of the metric.
        """
        with pytest.raises(TypeError):
            ScalarProductMetric(self.space.metric, scale)

    def test_scale_composition(self, scale_a, scale_b):
        """Check scaling a scaled metric multiplies scales instead of nesting.

        Parameters
        ----------
        scale_a : float
            Scale of the metric the space is equipped with.
        scale_b : float
            Scale applied to the scaled metric.
        """
        space = copy.copy(self.space).equip_with_metric(
            ScalarProductMetric, scale=scale_a
        )
        scaled_metric = scale_b * space.metric

        self.assertAllClose(scaled_metric.scale, scale_a * scale_b)
        assert scaled_metric.underlying_metric is self.space.metric


class ScaledMethodRegistrationTestCase(TestCase):
    def test_register_scaled_method_invalid(self, func_name, scaling_type):
        """Check registering an invalid scaled method raises.

        Parameters
        ----------
        func_name : str
            Name of the method.
        scaling_type : str
            Scaling type.
        """
        with pytest.raises(ValueError):
            register_scaled_method(func_name, scaling_type)

    def test_register_scaled_method_valid(self, scale):
        """Check a registered method is scaled.

        Parameters
        ----------
        scale : float
            Scale of the metric.
        """
        def dummy_method(self):
            return 1

        self.Metric.dummy_method = dummy_method
        register_scaled_method("dummy_method", "linear")
        scaled_metric_1 = scale * self.space.metric
        self.assertAllClose(scale, scaled_metric_1.dummy_method())
