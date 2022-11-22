"""Unit tests for the Siegel manifold."""

import geomstats.backend as gs
from tests.conftest import Parametrizer
from tests.data.siegel_data import SiegelMetricTestData, SiegelTestData
from tests.geometry_test_cases import ComplexRiemannianMetricTestCase, OpenSetTestCase


class TestSiegel(OpenSetTestCase, metaclass=Parametrizer):
    """Test of Siegel methods."""

    testing_data = SiegelTestData()

    def test_belongs(self, n, mat, expected):
        self.assertAllClose(self.Space(n).belongs(gs.array(mat)), expected)

    def test_projection(self, n, mat, expected):
        self.assertAllClose(
            self.Space(n).projection(gs.array(mat)),
            gs.array(expected),
        )


class TestSiegelMetric(ComplexRiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_ladder_parallel_transport = True
    skip_test_geodesic_bvp_belongs = True
    skip_test_geodesic_ivp_belongs = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_inner_product_is_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = SiegelMetricTestData()

    def test_inner_product(
        self, n, scale, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        metric = self.Metric(n, scale)
        result = metric.inner_product(
            gs.array(tangent_vec_a),
            gs.array(tangent_vec_b),
            gs.array(base_point),
        )
        self.assertAllClose(result, expected)

    def test_exp(self, n, scale, tangent_vec, base_point, expected):
        metric = self.Metric(n, scale)
        self.assertAllClose(
            metric.exp(
                gs.array(tangent_vec),
                gs.array(base_point),
            ),
            gs.array(expected),
        )

    def test_log(self, n, scale, point, base_point, expected):
        metric = self.Metric(n, scale)
        self.assertAllClose(
            metric.log(
                gs.array(point),
                gs.array(base_point),
            ),
            gs.array(expected),
        )

    def test_sectional_curvature(
        self, n, scale, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        metric = self.Metric(n, scale)
        result = metric.sectional_curvature(
            gs.array(tangent_vec_a),
            gs.array(tangent_vec_b),
            gs.array(base_point),
        )
        self.assertAllClose(result, expected)
