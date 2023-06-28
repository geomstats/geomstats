"""Unit tests for the Hermitian space."""


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
        self.assertAllClose(self.Space(dim).belongs(vec), expected)


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

    def test_exp(self, space, tangent_vec, base_point, expected):
        space.equip_with_metric(self.Metric)
        self.assertAllClose(space.metric.exp(tangent_vec, base_point), expected)

    def test_log(self, space, point, base_point, expected):
        space.equip_with_metric(self.Metric)
        self.assertAllClose(space.metric.log(point, base_point), expected)

    def test_inner_product(self, space, tangent_vec_a, tangent_vec_b, expected):
        space.equip_with_metric(self.Metric)
        self.assertAllClose(
            space.metric.inner_product(tangent_vec_a, tangent_vec_b),
            expected,
        )

    def test_squared_norm(self, space, vec, expected):
        space.equip_with_metric(self.Metric)
        self.assertAllClose(space.metric.squared_norm(vec), expected)

    def test_norm(self, space, vec, expected):
        space.equip_with_metric(self.Metric)
        self.assertAllClose(space.metric.norm(vec), expected)

    def test_metric_matrix(self, space, expected):
        space.equip_with_metric(self.Metric)
        self.assertAllClose(space.metric.metric_matrix(), expected)

    def test_squared_dist(self, space, point_a, point_b, expected):
        space.equip_with_metric(self.Metric)
        result = space.metric.squared_dist(point_a, point_b)
        self.assertAllClose(result, expected)

    def test_dist(self, space, point_a, point_b, expected):
        space.equip_with_metric(self.Metric)
        result = space.metric.dist(point_a, point_b)
        self.assertAllClose(result, expected)
