"""Unit tests for the Hermitian space."""


import geomstats.backend as gs
from tests.conftest import Parametrizer
from tests.data.hermitian_data import HermitianMetricTestData, HermitianTestData
from tests.geometry_test_cases import (
    ComplexRiemannianMetricTestCase,
    VectorSpaceTestCase,
)


class TestHermitian(VectorSpaceTestCase, metaclass=Parametrizer):
    skip_test_basis_belongs = True
    skip_test_basis_cardinality = True

    testing_data = HermitianTestData()

    def test_belongs(self, dim, vec, expected):
        self.assertAllClose(self.Space(dim).belongs(gs.array(vec)), gs.array(expected))


class TestHermitianMetric(ComplexRiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_inner_product_is_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = HermitianMetricTestData()

    def test_exp(self, dim, tangent_vec, base_point, expected):
        metric = self.Metric(dim)
        self.assertAllClose(
            metric.exp(gs.array(tangent_vec), gs.array(base_point)), gs.array(expected)
        )

    def test_log(self, dim, point, base_point, expected):
        metric = self.Metric(dim)
        self.assertAllClose(
            metric.log(gs.array(point), gs.array(base_point)), gs.array(expected)
        )

    def test_inner_product(self, dim, tangent_vec_a, tangent_vec_b, expected):
        metric = self.Metric(dim)
        self.assertAllClose(
            metric.inner_product(gs.array(tangent_vec_a), gs.array(tangent_vec_b)),
            gs.array(expected),
        )

    def test_squared_norm(self, dim, vec, expected):
        metric = self.Metric(dim)
        self.assertAllClose(metric.squared_norm(gs.array(vec)), gs.array(expected))

    def test_norm(self, dim, vec, expected):
        metric = self.Metric(dim)
        self.assertAllClose(metric.norm(gs.array(vec)), gs.array(expected))

    def test_metric_matrix(self, dim, expected):
        self.assertAllClose(self.Metric(dim).metric_matrix(), gs.array(expected))

    def test_squared_dist(self, dim, point_a, point_b, expected):
        metric = self.Metric(dim)
        result = metric.squared_dist(point_a, point_b)
        self.assertAllClose(result, gs.array(expected))

    def test_dist(self, dim, point_a, point_b, expected):
        metric = self.Metric(dim)
        result = metric.dist(point_a, point_b)
        self.assertAllClose(result, gs.array(expected))
