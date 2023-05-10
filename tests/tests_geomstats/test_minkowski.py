"""Unit tests for Minkowski space."""

from tests.conftest import Parametrizer, np_backend
from tests.data.minkowski_data import MinkowskiMetricTestData, MinkowskiTestData
from tests.geometry_test_cases import RiemannianMetricTestCase, VectorSpaceTestCase


class TestMinkowski(VectorSpaceTestCase, metaclass=Parametrizer):
    skip_test_basis_belongs = True
    skip_test_basis_cardinality = True

    testing_data = MinkowskiTestData()

    def test_belongs(self, dim, point, expected):
        self.assertAllClose(self.Space(dim).belongs(point), expected)


class TestMinkowskiMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_dist_is_positive = True
    skip_test_squared_dist_is_positive = True
    skip_test_dist_is_norm_of_log = not np_backend()
    skip_test_dist_is_symmetric = not np_backend()
    skip_test_triangle_inequality_of_dist = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = MinkowskiMetricTestData()

    def test_metric_matrix(self, space, expected):
        space.equip_with_metric(self.Metric)
        self.assertAllClose(space.metric.metric_matrix(), expected)

    def test_inner_product(self, space, point_a, point_b, expected):
        space.equip_with_metric(self.Metric)
        self.assertAllClose(
            space.metric.inner_product(point_a, point_b),
            expected,
        )

    def test_squared_norm(self, space, point, expected):
        space.equip_with_metric(self.Metric)
        self.assertAllClose(space.metric.squared_norm(point), expected)

    def test_exp(self, space, tangent_vec, base_point, expected):
        space.equip_with_metric(self.Metric)
        result = space.metric.exp(tangent_vec, base_point)
        self.assertAllClose(result, expected)

    def test_log(self, space, point, base_point, expected):
        space.equip_with_metric(self.Metric)
        result = space.metric.log(point, base_point)
        self.assertAllClose(result, expected)

    def test_squared_dist(self, space, point_a, point_b, expected):
        space.equip_with_metric(self.Metric)
        result = space.metric.squared_dist(point_a, point_b)
        self.assertAllClose(result, expected)
