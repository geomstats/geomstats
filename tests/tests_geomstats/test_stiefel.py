"""Unit tests for Stiefel manifolds."""

import pytest

import geomstats.backend as gs
from tests.conftest import Parametrizer, np_and_autograd_only
from tests.data.stiefel_data import StiefelCanonicalMetricTestData, StiefelTestData
from tests.geometry_test_cases import LevelSetTestCase, RiemannianMetricTestCase


class TestStiefel(LevelSetTestCase, metaclass=Parametrizer):
    skip_test_intrinsic_after_extrinsic = True
    skip_test_extrinsic_after_intrinsic = True
    skip_test_to_tangent_is_tangent = True
    skip_test_triangle_inequality_of_dist = True
    skip_test_squared_dist_is_positive = True

    testing_data = StiefelTestData()

    def test_to_grassmannian(self, point, expected):
        self.assertAllClose(self.Space.to_grassmannian(point), expected)


class TestStiefelCanonicalMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_after_log = True
    skip_test_log_after_exp = True
    skip_test_log_is_tangent = True
    skip_test_geodesic_bvp_belongs = True
    skip_test_squared_dist_is_symmetric = True
    skip_test_exp_shape = True
    skip_test_log_shape = True
    skip_test_exp_ladder_parallel_transport = True
    skip_test_dist_is_symmetric = True
    skip_test_dist_is_norm_of_log = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True
    skip_test_squared_dist_is_positive = True
    skip_test_triangle_inequality_of_dist = True
    skip_test_dist_is_positive = True

    testing_data = StiefelCanonicalMetricTestData()

    def test_log_two_sheets_error(self, space, point, base_point, expected):
        space.equip_with_metric(self.Metric)
        with expected:
            space.metric.log(point, base_point)

    @pytest.mark.skip(reason="throwing value error")
    def test_retraction_lifting(self, space, tangent_vec, base_point, rtol, atol):
        space.equip_with_metric(self.Metric)
        lifted = space.metric.lifting(tangent_vec, base_point)
        result = space.metric.retraction(lifted, base_point)
        self.assertAllClose(result, tangent_vec, rtol, atol)

    @pytest.mark.skip(reason="throwing value error")
    def test_lifting_retraction(self, space, point, base_point, rtol, atol):
        space.equip_with_metric(self.Metric)
        retract = space.metric.retraction(point, base_point)
        result = space.metric.lifting(retract, base_point)
        self.assertAllClose(result, point, rtol, atol)

    @pytest.mark.skip(reason="throwing value error")
    def test_lifting_shape(self, space, point, base_point, expected):
        space.equip_with_metric(self.Metric)
        result = space.metric.lifting(point, base_point)
        self.assertAllClose(gs.shape(result), expected)

    @np_and_autograd_only
    def test_retraction_shape(self, space, tangent_vec, base_point, expected):
        space.equip_with_metric(self.Metric)
        result = space.metric.retraction(tangent_vec, base_point)
        self.assertAllClose(gs.shape(result), expected)
