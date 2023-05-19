"""Unit tests for the Euclidean space."""

from geomstats.geometry.euclidean import Euclidean
from tests.conftest import Parametrizer
from tests.data.euclidean_data import EuclideanMetricTestData, EuclideanTestData
from tests.geometry_test_cases import RiemannianMetricTestCase, VectorSpaceTestCase


class TestEuclidean(VectorSpaceTestCase, metaclass=Parametrizer):
    skip_test_basis_belongs = True
    skip_test_basis_cardinality = True

    testing_data = EuclideanTestData()

    def test_belongs(self, dim, vec, expected):
        self.assertAllClose(self.Space(dim).belongs(vec), expected)


class TestEuclideanMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_exp_geodesic_ivp = True
    skip_test_riemann_tensor_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_sectional_curvature_shape = True
    testing_data = EuclideanMetricTestData()

    def test_exp(self, dim, tangent_vec, base_point, expected):
        space = Euclidean(dim)
        self.assertAllClose(space.metric.exp(tangent_vec, base_point), expected)

    def test_log(self, dim, point, base_point, expected):
        space = Euclidean(dim)
        self.assertAllClose(space.metric.log(point, base_point), expected)

    def test_inner_product(self, dim, tangent_vec_a, tangent_vec_b, expected):
        space = Euclidean(dim)
        self.assertAllClose(
            space.metric.inner_product(tangent_vec_a, tangent_vec_b),
            expected,
        )

    def test_squared_norm(self, dim, vec, expected):
        space = Euclidean(dim)
        self.assertAllClose(space.metric.squared_norm(vec), expected)

    def test_norm(self, dim, vec, expected):
        space = Euclidean(dim)
        self.assertAllClose(space.metric.norm(vec), expected)

    def test_metric_matrix(self, dim, expected):
        space = Euclidean(dim)
        self.assertAllClose(space.metric.metric_matrix(), expected)

    def test_squared_dist(self, dim, point_a, point_b, expected):
        space = Euclidean(dim)
        result = space.metric.squared_dist(point_a, point_b)
        self.assertAllClose(result, expected)

    def test_dist(self, dim, point_a, point_b, expected):
        space = Euclidean(dim)
        result = space.metric.dist(point_a, point_b)
        self.assertAllClose(result, expected)
