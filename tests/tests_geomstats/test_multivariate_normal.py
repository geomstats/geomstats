"""Unit tests for the MultivariateDiagonalNormalDistributions manifold."""

from scipy.stats import multivariate_normal

import geomstats.backend as gs
from tests.conftest import Parametrizer
from tests.data.multivariate_normal import (
    MultivariateDiagonalNormalDistributionsTestData,
    MultivariateDiagonalNormalMetricTestData,
)
from tests.geometry_test_cases import OpenSetTestCase, RiemannianMetricTestCase


class TestMultivariateDiagonalNormalDistributions(
    OpenSetTestCase, metaclass=Parametrizer
):
    testing_data = MultivariateDiagonalNormalDistributionsTestData()

    def test_belongs(self, n, point, expected):
        self.assertAllClose(self.Space(n).belongs(point), expected)

    def test_random_point_shape(self, point, expected):
        self.assertAllClose(point.shape, expected)

    def test_sample(self, n, point, n_samples, expected):
        self.assertAllClose(self.Space(n).sample(point, n_samples).shape, expected)

    def test_point_to_pdf(self, n, point, n_samples):
        point = gs.to_ndarray(point, to_ndim=2, axis=0)
        Space = self.Space(n)
        samples = Space.sample(point[0, :], n_samples)
        samples = gs.to_ndarray(samples, to_ndim=2, axis=0)

        expected = []
        for i in range(point.shape[0]):
            loc, cov = Space._unstack_location_diagonal(point[i, ...])
            tmp = list()
            for j in range(samples.shape[0]):
                x = samples[j, ...]
                tmp.append(multivariate_normal.pdf(x, mean=loc, cov=cov))
            expected.append(gs.array(tmp))
        expected = gs.squeeze(gs.stack(expected, axis=0))
        self.assertAllClose(Space.pdf(samples, point), expected)


class TestMultivariateDiagonalNormalMetric(
    RiemannianMetricTestCase, metaclass=Parametrizer
):
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_geodesic_ivp_belongs = True
    skip_test_geodesic_bvp_belongs = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_ladder_parallel_transport = True
    skip_test_riemann_tensor_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_sectional_curvature_shape = True

    testing_data = MultivariateDiagonalNormalMetricTestData()
    Space = testing_data.Space
