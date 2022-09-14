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
    testing_data = EuclideanMetricTestData()

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
