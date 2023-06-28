"""Unit tests for Positive lower triangular matrices"""

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from tests.conftest import Parametrizer
from tests.data.positive_lower_triangular_matrices_data import (
    CholeskyMetricTestData,
    PositiveLowerTriangularMatricesTestData,
)
from tests.geometry_test_cases import OpenSetTestCase, RiemannianMetricTestCase


class TestPositiveLowerTriangularMatrices(OpenSetTestCase, metaclass=Parametrizer):
    testing_data = PositiveLowerTriangularMatricesTestData()

    def test_belongs(self, n, mat, expected):
        self.assertAllClose(self.Space(n).belongs(mat), expected)

    def test_gram(self, n, point, expected):
        self.assertAllClose(self.Space(n).gram(point), expected)

    def test_differential_gram(self, n, tangent_vec, base_point, expected):
        self.assertAllClose(
            self.Space(n).differential_gram(tangent_vec, base_point),
            expected,
        )

    def test_inverse_differential_gram(self, n, tangent_vec, base_point, expected):
        self.assertAllClose(
            self.Space(n).inverse_differential_gram(tangent_vec, base_point),
            expected,
        )

    @tests.conftest.np_and_autograd_only
    def test_differential_gram_belongs(self, n, tangent_vec, base_point):
        result = self.Space(n).differential_gram(tangent_vec, base_point)
        self.assertAllClose(gs.all(SymmetricMatrices(n).belongs(result)), True)

    def test_inverse_differential_gram_belongs(self, n, tangent_vec, base_point):
        result = self.Space(n).inverse_differential_gram(tangent_vec, base_point)
        self.assertAllClose(gs.all(self.Space(n).embedding_space.belongs(result)), True)


class TestCholeskyMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = CholeskyMetricTestData()

    def test_diag_inner_product(
        self, space, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        space.equip_with_metric(self.Metric)
        result = space.metric.diag_inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(result, expected)

    def test_strictly_lower_inner_product(
        self, space, tangent_vec_a, tangent_vec_b, expected
    ):
        space.equip_with_metric(self.Metric)
        result = space.metric.strictly_lower_inner_product(tangent_vec_a, tangent_vec_b)
        self.assertAllClose(result, expected)

    def test_inner_product(
        self, space, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        space.equip_with_metric(self.Metric)
        result = space.metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(result, expected)

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
