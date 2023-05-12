"""Unit tests for the geometric manifold."""

from scipy.stats import geom

import geomstats.backend as gs
import tests.conftest
from tests.conftest import Parametrizer, np_backend
from tests.data.geometric_data import GeometricMetricTestData, GeometricTestData
from tests.geometry_test_cases import OpenSetTestCase, RiemannianMetricTestCase

NOT_AUTODIFF = np_backend()


class TestGeometric(OpenSetTestCase, metaclass=Parametrizer):
    testing_data = GeometricTestData()

    def test_belongs(self, point, expected):
        self.assertAllClose(self.Space().belongs(point), expected)

    def test_random_point_shape(self, point, expected):
        self.assertAllClose(point.shape, expected)

    def test_sample_shape(self, point, n_samples, expected):
        self.assertAllClose(self.Space().sample(point, n_samples).shape, expected)

    @tests.conftest.np_and_autograd_only
    def test_point_to_pdf(self, point, n_samples):
        space = self.Space()
        point = gs.to_ndarray(point, 1)
        n_points = point.shape[0]
        pmf = space.point_to_pdf(point)
        point_to_sample = point[0] if point.ndim > 1 else point
        samples = gs.to_ndarray(space.sample(point_to_sample, n_samples), 1)
        result = gs.squeeze(pmf(samples))
        pmf = []
        for i in range(n_points):
            pmf.append(gs.array([geom.pmf(x, point[i]) for x in samples]))
        expected = gs.squeeze(gs.stack(pmf, axis=0))
        self.assertAllClose(result, expected)


class TestGeometricMetric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_ladder_parallel_transport = True
    skip_test_riemann_tensor_shape = NOT_AUTODIFF
    skip_test_ricci_tensor_shape = NOT_AUTODIFF
    skip_test_scalar_curvature_shape = NOT_AUTODIFF
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = NOT_AUTODIFF
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = NOT_AUTODIFF
    skip_test_covariant_riemann_tensor_bianchi_identity = NOT_AUTODIFF
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = NOT_AUTODIFF
    skip_test_sectional_curvature_shape = NOT_AUTODIFF
    testing_data = GeometricMetricTestData()
    Space = testing_data.Space

    def test_squared_dist(self, space, point_a, point_b, expected):
        space.equip_with_metric(self.Metric)
        self.assertAllClose(space.metric.squared_dist(point_a, point_b), expected)

    def test_metric_matrix(self, space, point, expected):
        space.equip_with_metric(self.Metric)
        self.assertAllClose(space.metric.metric_matrix(point), expected)

    def test_geodesic_symmetry(self, space):
        space.equip_with_metric(self.Metric)
        point_a, point_b = space.random_point(2)
        path_ab = space.metric.geodesic(initial_point=point_a, end_point=point_b)
        path_ba = space.metric.geodesic(initial_point=point_b, end_point=point_a)
        t = gs.linspace(0.0, 1.0, 10)
        self.assertAllClose(path_ab(t), path_ba(1 - t))
