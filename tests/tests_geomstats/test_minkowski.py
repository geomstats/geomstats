"""Unit tests for Minkowski space."""

import geomstats.backend as gs
from tests.conftest import Parametrizer, np_backend
from tests.data.minkowski_data import MinkowskiMetricTestData, MinkowskiTestData
from tests.geometry_test_cases import RiemannianMetricTestCase, VectorSpaceTestCase


class TestMinkowski(VectorSpaceTestCase, metaclass=Parametrizer):
    skip_test_basis_belongs = True
    skip_test_basis_cardinality = True

    testing_data = MinkowskiTestData()

    def test_belongs(self, dim, point, expected):
        self.assertAllClose(
            self.Space(dim).belongs(gs.array(point)), gs.array(expected)
        )


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

    def test_metric_matrix(self, dim, expected):
        space = self.Space(dim)
        self.assertAllClose(space.metric.metric_matrix(), gs.array(expected))

    def test_inner_product(self, dim, point_a, point_b, expected):
        space = self.Space(dim)
        self.assertAllClose(
            space.metric.inner_product(gs.array(point_a), gs.array(point_b)),
            gs.array(expected),
        )

    def test_squared_norm(self, dim, point, expected):
        space = self.Space(dim)
        self.assertAllClose(space.metric.squared_norm(gs.array(point)), gs.array(expected))

    def test_exp(self, dim, tangent_vec, base_point, expected):
        result = self.Space(dim).metric.exp(gs.array(tangent_vec), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))

    def test_log(self, dim, point, base_point, expected):
        result = self.Space(dim).metric.log(gs.array(point), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))

    def test_squared_dist(self, dim, point_a, point_b, expected):
        result = self.Space(dim).metric.squared_dist(gs.array(point_a), gs.array(point_b))
        self.assertAllClose(result, gs.array(expected))
