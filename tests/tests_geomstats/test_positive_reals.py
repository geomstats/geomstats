"""Unit tests for the manifold of positive reals."""

import geomstats.backend as gs
from tests.conftest import Parametrizer
from tests.data.positive_reals_data import (
    PositiveRealsMetricTestData,
    PositiveRealsTestData,
)
from tests.geometry_test_cases import OpenSetTestCase, RiemannianMetricTestCase


class TestPositiveReals(OpenSetTestCase, metaclass=Parametrizer):
    """Test of PositiveReals methods."""

    testing_data = PositiveRealsTestData()

    def test_belongs(self, point, expected):
        self.assertAllClose(self.Space().belongs(gs.array(point)), gs.array(expected))

    def test_projection(self, point, expected):
        self.assertAllClose(
            self.Space().projection(gs.array(point)), gs.array(expected)
        )


class TestPositiveRealsMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_ladder_parallel_transport = True
    skip_test_geodesic_ivp_belongs = True
    skip_test_geodesic_bvp_belongs = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = PositiveRealsMetricTestData()

    def test_inner_product(self, tangent_vec_a, tangent_vec_b, base_point, expected):
        metric = self.Metric()
        result = metric.inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b), gs.array(base_point)
        )
        self.assertAllClose(result, expected)

    def test_exp(self, tangent_vec, base_point, expected):
        metric = self.Metric()
        self.assertAllClose(
            metric.exp(gs.array(tangent_vec), gs.array(base_point)), gs.array(expected)
        )

    def test_log(self, point, base_point, expected):
        metric = self.Metric()
        self.assertAllClose(
            metric.log(gs.array(point), gs.array(base_point)), gs.array(expected)
        )
