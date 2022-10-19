"""Unit tests for the Grassmannian."""

import geomstats.backend as gs
from tests.conftest import Parametrizer, np_backend
from tests.data.grassmannian_data import (
    GrassmannianCanonicalMetricTestData,
    GrassmannianTestData,
)
from tests.geometry_test_cases import LevelSetTestCase, RiemannianMetricTestCase


class TestGrassmannian(LevelSetTestCase, metaclass=Parametrizer):
    skip_test_intrinsic_after_extrinsic = True
    skip_test_extrinsic_after_intrinsic = True

    testing_data = GrassmannianTestData()

    def test_belongs(self, n, p, point, expected):
        self.assertAllClose(self.Space(n, p).belongs(point), gs.array(expected))


class TestGrassmannianCanonicalMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_log_after_exp = True
    skip_test_exp_geodesic_ivp = True
    skip_test_log_is_tangent = not np_backend()
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = GrassmannianCanonicalMetricTestData()

    def test_exp(self, n, p, tangent_vec, base_point, expected):
        self.assertAllClose(
            self.Metric(n, p).exp(gs.array(tangent_vec), gs.array(base_point)),
            gs.array(expected),
        )
