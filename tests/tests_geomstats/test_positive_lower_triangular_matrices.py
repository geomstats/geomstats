"""Unit tests for Positive lower triangular matrices"""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.positive_lower_triangular_matrices import (
    CholeskyMetric,
    PositiveLowerTriangularMatrices,
)
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from tests.conftest import Parametrizer
from tests.data.positive_lower_triangular_matrices_data import (
    CholeskyMetricTestData,
    PositiveLowerTriangularMatricesTestData,
)
from tests.geometry_test_cases import OpenSetTestCase, RiemannianMetricTestCase


class TestPositiveLowerTriangularMatrices(OpenSetTestCase, metaclass=Parametrizer):
    """Test of Cholesky methods."""

    space = PositiveLowerTriangularMatrices

    testing_data = PositiveLowerTriangularMatricesTestData()

    def test_belongs(self, n, mat, expected):
        self.assertAllClose(self.space(n).belongs(gs.array(mat)), gs.array(expected))

    def test_gram(self, n, point, expected):
        self.assertAllClose(self.space(n).gram(gs.array(point)), gs.array(expected))

    def test_differential_gram(self, n, tangent_vec, base_point, expected):
        self.assertAllClose(
            self.space(n).differential_gram(
                gs.array(tangent_vec), gs.array(base_point)
            ),
            gs.array(expected),
        )

    def test_inverse_differential_gram(self, n, tangent_vec, base_point, expected):
        self.assertAllClose(
            self.space(n).inverse_differential_gram(
                gs.array(tangent_vec), gs.array(base_point)
            ),
            gs.array(expected),
        )

    @geomstats.tests.np_and_autograd_only
    def test_differential_gram_belongs(self, n, tangent_vec, base_point):
        result = self.space(n).differential_gram(
            gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(gs.all(SymmetricMatrices(n).belongs(result)), True)

    def test_inverse_differential_gram_belongs(self, n, tangent_vec, base_point):
        result = self.space(n).inverse_differential_gram(
            gs.array(tangent_vec), gs.array(base_point)
        )
        self.assertAllClose(gs.all(self.space(n).ambient_space.belongs(result)), True)


class TestCholeskyMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    metric = connection = CholeskyMetric
    space = PositiveLowerTriangularMatrices

    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True

    testing_data = CholeskyMetricTestData()

    def test_diag_inner_product(
        self, n, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        result = self.metric(n).diag_inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_strictly_lower_inner_product(
        self, n, tangent_vec_a, tangent_vec_b, expected
    ):
        result = self.metric(n).strictly_lower_inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_inner_product(self, n, tangent_vec_a, tangent_vec_b, base_point, expected):
        result = self.metric(n).inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b), gs.array(base_point)
        )
        self.assertAllClose(result, gs.array(expected))

    def test_exp(self, n, tangent_vec, base_point, expected):
        result = self.metric(n).exp(gs.array(tangent_vec), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))

    def test_log(self, n, point, base_point, expected):
        result = self.metric(n).log(gs.array(point), gs.array(base_point))
        self.assertAllClose(result, gs.array(expected))

    def test_squared_dist(self, n, point_a, point_b, expected):
        result = self.metric(n).squared_dist(gs.array(point_a), gs.array(point_b))
        self.assertAllClose(result, gs.array(expected))
