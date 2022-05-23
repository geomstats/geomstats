"""Unit tests for Stiefel manifolds."""

import pytest

import geomstats.backend as gs
from tests.conftest import Parametrizer, np_autograd_and_tf_only
from tests.data.stiefel_data import StiefelCanonicalMetricTestData, StiefelTestData
from tests.geometry_test_cases import LevelSetTestCase, RiemannianMetricTestCase


class TestStiefel(LevelSetTestCase, metaclass=Parametrizer):
    skip_test_intrinsic_after_extrinsic = True
    skip_test_extrinsic_after_intrinsic = True
    skip_test_to_tangent_is_tangent = True

    testing_data = StiefelTestData()
    space = testing_data.space

    def test_to_grassmannian(self, point, expected):
        self.assertAllClose(
            self.space.to_grassmannian(gs.array(point)), gs.array(expected)
        )


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

    testing_data = StiefelCanonicalMetricTestData()
    Metric = Connection = testing_data.Metric

    def test_log_two_sheets_error(self, n, p, point, base_point, expected):
        metric = self.Metric(n, p)
        with expected:
            metric.log(point, base_point)

    @pytest.mark.skip(reason="throwing value error")
    def test_retraction_lifting(
        self, connection_args, tangent_vec, base_point, rtol, atol
    ):
        metric = self.Metric(*connection_args)
        lifted = metric.lifting(gs.array(tangent_vec), gs.array(base_point))
        result = metric.retraction(lifted, gs.array(base_point))
        self.assertAllClose(result, gs.array(tangent_vec), rtol, atol)

    @pytest.mark.skip(reason="throwing value error")
    def test_lifting_retraction(self, connection_args, point, base_point, rtol, atol):
        metric = self.Metric(*connection_args)
        retract = metric.retraction(gs.array(point), gs.array(base_point))
        result = metric.lifting(retract, gs.array(base_point))
        self.assertAllClose(result, gs.array(point), rtol, atol)

    @pytest.mark.skip(reason="throwing value error")
    def test_lifting_shape(self, connection_args, point, base_point, expected):
        metric = self.Metric(*connection_args)
        result = metric.lifting(gs.array(point), gs.array(base_point))
        self.assertAllClose(gs.shape(result), expected)

    @np_autograd_and_tf_only
    def test_retraction_shape(self, connection_args, tangent_vec, base_point, expected):
        metric = self.Metric(*connection_args)
        result = metric.retraction(gs.array(tangent_vec), gs.array(base_point))
        self.assertAllClose(gs.shape(result), expected)
