"""Unit tests for the Hermitian space."""


import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hermitian import Hermitian, HermitianMetric
from geomstats.tests import tf_backend
from tests.conftest import Parametrizer
from tests.data.hermitian_data import HermitianMetricTestData, HermitianTestData
from tests.geometry_test_cases import RiemannianMetricTestCase, VectorSpaceTestCase


class TestHermitian(VectorSpaceTestCase, metaclass=Parametrizer):
    space = Hermitian
    skip_test_basis_belongs = True
    skip_test_basis_cardinality = True
    skip_test_belongs = tf_backend()

    testing_data = HermitianTestData()

    def test_belongs(self, dim, vec, expected):
        self.assertAllClose(self.space(dim).belongs(gs.array(vec)), gs.array(expected))


class TestHermitianMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    metric = connection = HermitianMetric
    skip_test_exp = tf_backend()
    skip_test_log = tf_backend()
    skip_test_inner_product = tf_backend()
    skip_test_dist = geomstats.tests.tf_backend()
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True

    testing_data = HermitianMetricTestData()

    def test_exp(self, dim, tangent_vec, base_point, expected):
        metric = HermitianMetric(dim)
        self.assertAllClose(
            metric.exp(gs.array(tangent_vec), gs.array(base_point)), gs.array(expected)
        )

    def test_log(self, dim, point, base_point, expected):
        metric = HermitianMetric(dim)
        self.assertAllClose(
            metric.log(gs.array(point), gs.array(base_point)), gs.array(expected)
        )

    def test_inner_product(self, dim, tangent_vec_a, tangent_vec_b, expected):
        metric = HermitianMetric(dim)
        self.assertAllClose(
            metric.inner_product(gs.array(tangent_vec_a), gs.array(tangent_vec_b)),
            gs.array(expected),
        )

    def test_squared_norm(self, dim, vec, expected):
        metric = HermitianMetric(dim)
        self.assertAllClose(metric.squared_norm(gs.array(vec)), gs.array(expected))

    def test_norm(self, dim, vec, expected):
        metric = HermitianMetric(dim)
        self.assertAllClose(metric.norm(gs.array(vec)), gs.array(expected))

    def test_metric_matrix(self, dim, expected):
        self.assertAllClose(HermitianMetric(dim).metric_matrix(), gs.array(expected))

    def test_squared_dist(self, dim, point_a, point_b, expected):
        metric = HermitianMetric(dim)
        result = metric.squared_dist(point_a, point_b)
        self.assertAllClose(result, gs.array(expected))

    def test_dist(self, dim, point_a, point_b, expected):
        metric = HermitianMetric(dim)
        result = metric.dist(point_a, point_b)
        self.assertAllClose(result, gs.array(expected))
