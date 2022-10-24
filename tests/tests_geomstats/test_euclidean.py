"""Unit tests for the Euclidean space."""

import geomstats.backend as gs
from tests.conftest import Parametrizer
from tests.data.euclidean_data import EuclideanMetricTestData, EuclideanTestData
from tests.geometry_test_cases import RiemannianMetricTestCase, VectorSpaceTestCase


class TestEuclidean(VectorSpaceTestCase, metaclass=Parametrizer):
    skip_test_basis_belongs = True
    skip_test_basis_cardinality = True

    testing_data = EuclideanTestData()

    def test_belongs(self, dim, vec, expected):
        self.assertAllClose(self.Space(dim).belongs(gs.array(vec)), gs.array(expected))


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
        space = self.Space(dim)
        self.assertAllClose(
            space.metric.exp(gs.array(tangent_vec), gs.array(base_point)), gs.array(expected)
        )

    def test_log(self, dim, point, base_point, expected):
        space = self.Space(dim)
        self.assertAllClose(
            space.metric.log(gs.array(point), gs.array(base_point)), gs.array(expected)
        )

    def test_inner_product(self, dim, tangent_vec_a, tangent_vec_b, expected):
        space = self.Space(dim)
        self.assertAllClose(
            space.metric.inner_product(gs.array(tangent_vec_a), gs.array(tangent_vec_b)),
            gs.array(expected),
        )

    def test_squared_norm(self, dim, vec, expected):
        space = self.Space(dim)
        self.assertAllClose(space.metric.squared_norm(gs.array(vec)), gs.array(expected))

    def test_norm(self, dim, vec, expected):
        space = self.Space(dim)
        self.assertAllClose(space.metric.norm(gs.array(vec)), gs.array(expected))

    def test_metric_matrix(self, dim, expected):
        self.assertAllClose(self.Space(dim).metric.metric_matrix(), gs.array(expected))

    def test_squared_dist(self, dim, point_a, point_b, expected):
        space = self.Space(dim)
        result = space.metric.squared_dist(point_a, point_b)
        self.assertAllClose(result, gs.array(expected))

    def test_dist(self, dim, point_a, point_b, expected):
        space = self.Space(dim)
        result = space.metric.dist(point_a, point_b)
        self.assertAllClose(result, gs.array(expected))
