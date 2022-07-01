"""Unit tests for the Grassmannian."""

import geomstats.backend as gs
from geomstats.geometry.grassmannian import Grassmannian, GrassmannianCanonicalMetric
from geomstats.tests import np_backend
from tests.conftest import Parametrizer
from tests.data.grassmannian_data import (
    GrassmannianCanonicalMetricTestData,
    GrassmannianTestData,
)
from tests.geometry_test_cases import LevelSetTestCase, RiemannianMetricTestCase


class TestGrassmannian(LevelSetTestCase, metaclass=Parametrizer):
    space = Grassmannian
    skip_test_intrinsic_after_extrinsic = True
    skip_test_extrinsic_after_intrinsic = True

    testing_data = GrassmannianTestData()

    def test_belongs(self, n, p, point, expected):
        self.assertAllClose(self.space(n, p).belongs(point), gs.array(expected))


class TestGrassmannianCanonicalMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    metric = connection = GrassmannianCanonicalMetric
    skip_test_log_after_exp = True
    skip_test_exp_geodesic_ivp = True
    skip_test_log_is_tangent = not np_backend()

    testing_data = GrassmannianCanonicalMetricTestData()

    def test_exp(self, n, p, tangent_vec, base_point, expected):
        self.assertAllClose(
            self.metric(n, p).exp(gs.array(tangent_vec), gs.array(base_point)),
            gs.array(expected),
        )
