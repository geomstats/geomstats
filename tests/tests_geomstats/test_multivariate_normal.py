"""Unit tests for the MultivariateDiagonalNormalDistributions manifold."""

from scipy.stats import multivariate_normal

import geomstats.backend as gs
from tests.conftest import Parametrizer, tf_backend
from tests.data.multivariate_normal import (
    MultivariateCenteredNormalDistributionsTestData,
    MultivariateDiagonalNormalDistributionsTestData,
    MultivariateDiagonalNormalMetricTestData,
    MultivariateGeneralNormalDistributionsTestData,
)
from tests.geometry_test_cases import (
    ManifoldTestCase,
    OpenSetTestCase,
    RiemannianMetricTestCase,
)

TF_BACKEND = tf_backend()


class TestMultivariateCenteredNormalDistributions(
    OpenSetTestCase, metaclass=Parametrizer
):
    testing_data = MultivariateCenteredNormalDistributionsTestData()

    def test_belongs(self, n, point, expected):
        self.assertAllClose(self.Space(n).belongs(point), expected)

    def test_random_point_shape(self, point, expected):
        self.assertAllClose(point.shape, expected)

    def test_sample(self, n, point, n_samples, expected):
        self.assertAllClose(self.Space(n).sample(point, n_samples).shape, expected)

    def test_point_to_pdf(self, n, point, n_samples):
        space = self.Space(n)
        samples = space.sample(space.random_point(), n_samples)
        result = space.point_to_pdf(point)(samples)

        samples = gs.to_ndarray(samples, to_ndim=2, axis=0)
        point = gs.to_ndarray(point, to_ndim=3, axis=0)
        expected = []
        for i in range(point.shape[0]):
            tmp = list()
            loc, cov = gs.zeros(n), point[i]
            for j in range(samples.shape[0]):
                x = samples[j]
                tmp.append(multivariate_normal.pdf(x, mean=loc, cov=cov))
            expected.append(gs.array(tmp))
        expected = gs.transpose(gs.squeeze(gs.stack(expected, axis=0)))
        self.assertAllClose(result, expected)


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
        space = self.Space(n)
        samples = space.sample(space.random_point(), n_samples)
        result = space.point_to_pdf(point)(samples)

        samples = gs.to_ndarray(samples, to_ndim=2, axis=0)
        point = gs.to_ndarray(point, to_ndim=2, axis=0)
        expected = []
        for i in range(point.shape[0]):
            loc, cov = space._unstack_location_diagonal(n, point[i, ...])
            tmp = list()
            for j in range(samples.shape[0]):
                x = samples[j, ...]
                tmp.append(multivariate_normal.pdf(x, mean=loc, cov=cov))
            expected.append(gs.array(tmp))
        expected = gs.squeeze(gs.stack(expected, axis=0))
        self.assertAllClose(result, expected)


class TestMultivariateGeneralNormalDistributions(
    ManifoldTestCase, metaclass=Parametrizer
):
    testing_data = MultivariateGeneralNormalDistributionsTestData()
    skip_test_belongs = TF_BACKEND

    def test_belongs(self, n, point, expected):
        self.assertAllClose(self.Space(n).belongs(point), expected)

    def test_random_point_shape(self, point, expected):
        self.assertAllClose(point.shape, expected)

    def test_sample(self, n, point, n_samples, expected):
        self.assertAllClose(self.Space(n).sample(point, n_samples).shape, expected)

    def test_point_to_pdf(self, n, point, n_samples):
        space = self.Space(n)
        samples = space.sample(space.random_point(), n_samples)
        result = space.point_to_pdf(point)(samples)

        samples = gs.to_ndarray(samples, to_ndim=2, axis=0)
        point = gs.to_ndarray(point, to_ndim=2, axis=0)
        expected = []
        for i in range(point.shape[0]):
            tmp = list()
            loc, cov = space.reformat(point[i])
            for j in range(samples.shape[0]):
                x = samples[j]
                tmp.append(multivariate_normal.pdf(x, mean=loc, cov=cov))
            expected.append(gs.array(tmp))
        expected = gs.transpose(gs.squeeze(gs.stack(expected, axis=0)))
        self.assertAllClose(result, expected)


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

    def test_inner_product_shape(
        self, metric, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        result = metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        result = result.shape
        self.assertAllClose(result, expected)
