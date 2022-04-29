"""Unit tests for the Hypersphere."""

import scipy.special

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
from geomstats.learning.frechet_mean import FrechetMean
from tests.conftest import Parametrizer
from tests.data.hypersphere_data import HypersphereMetricTestData, HypersphereTestData
from tests.geometry_test_cases import LevelSetTestCase, RiemannianMetricTestCase

MEAN_ESTIMATION_TOL = 1e-1
KAPPA_ESTIMATION_TOL = 1e-1
ONLINE_KMEANS_TOL = 1e-1


class TestHypersphere(LevelSetTestCase, metaclass=Parametrizer):
    space = Hypersphere

    testing_data = HypersphereTestData()

    def test_replace_values(self, dim, points, new_points, indcs, expected):
        space = self.space(dim)
        result = space._replace_values(
            gs.array(points), gs.array(new_points), gs.array(indcs)
        )
        self.assertAllClose(result, expected)

    def test_angle_to_extrinsic(self, dim, point, expected):
        space = self.space(dim)
        result = space.angle_to_extrinsic(point)
        self.assertAllClose(result, expected)

    def test_extrinsic_to_angle(self, dim, point, expected):
        space = self.space(dim)
        result = space.extrinsic_to_angle(point)
        self.assertAllClose(result, expected)

    def test_spherical_to_extrinsic(self, dim, point, expected):
        space = self.space(dim)
        result = space.spherical_to_extrinsic(point)
        self.assertAllClose(result, expected)

    def test_extrinsic_to_spherical(self, dim, point, expected):
        space = self.space(dim)
        result = space.extrinsic_to_spherical(point)
        self.assertAllClose(result, expected)

    def test_random_von_mises_fisher_belongs(self, dim, n_samples):
        space = self.space(dim)
        result = space.belongs(space.random_von_mises_fisher(n_samples=n_samples))
        self.assertAllClose(gs.all(result), gs.array(True))

    def test_random_von_mises_fisher_mean(self, dim, kappa, n_samples, expected):
        space = self.space(dim)
        points = space.random_von_mises_fisher(kappa=kappa, n_samples=n_samples)
        sum_points = gs.sum(points, axis=0)
        result = sum_points / gs.linalg.norm(sum_points)
        self.assertAllClose(result, expected, atol=KAPPA_ESTIMATION_TOL)

    def test_tangent_spherical_to_extrinsic(
        self, dim, tangent_vec_spherical, base_point_spherical, expected
    ):
        space = self.space(dim)
        result = space.tangent_spherical_to_extrinsic(
            tangent_vec_spherical, base_point_spherical
        )
        self.assertAllClose(result, expected)

    def test_tangent_extrinsic_to_spherical(
        self, dim, tangent_vec, base_point, base_point_spherical, expected
    ):
        space = self.space(dim)
        result = space.tangent_extrinsic_to_spherical(
            tangent_vec, base_point, base_point_spherical
        )
        self.assertAllClose(result, expected)

    def test_tangent_extrinsic_to_spherical_raises(
        self, dim, tangent_vec, base_point, base_point_spherical, expected
    ):
        space = self.space(dim)
        with expected:
            space.tangent_extrinsic_to_spherical(
                tangent_vec, base_point, base_point_spherical
            )

    @geomstats.tests.np_autograd_and_torch_only
    def test_riemannian_normal_frechet_mean(self, dim):
        space = self.space(dim)
        mean = space.random_uniform()
        precision = gs.eye(space.dim) * 10
        sample = space.random_riemannian_normal(mean, precision, 30000)
        estimator = FrechetMean(space.metric, method="adaptive")
        estimator.fit(sample)
        estimate = estimator.estimate_
        self.assertAllClose(estimate, mean, atol=1e-1)

    @geomstats.tests.np_autograd_and_torch_only
    def test_riemannian_normal_and_belongs(self, dim, n_points):
        space = self.space(dim)
        mean = space.random_uniform()
        cov = gs.eye(dim)
        sample = space.random_riemannian_normal(mean, cov, n_points)
        result = space.belongs(sample)
        self.assertTrue(gs.all(result))

    def test_sample_von_mises_fisher_mean(self, dim, mean, kappa, n_points):
        """
        Check that the maximum likelihood estimates of the mean and
        concentration parameter are close to the real values. A first
        estimation of the concentration parameter is obtained by a
        closed-form expression and improved through the Newton method.
        """
        space = self.space(dim)
        points = space.random_von_mises_fisher(mu=mean, kappa=kappa, n_samples=n_points)
        sum_points = gs.sum(points, axis=0)
        result = sum_points / gs.linalg.norm(sum_points)
        expected = mean
        self.assertAllClose(result, expected, atol=MEAN_ESTIMATION_TOL)

    def test_sample_random_von_mises_fisher_kappa(self, dim, kappa, n_points):
        # check concentration parameter for dispersed distribution
        sphere = Hypersphere(dim)
        points = sphere.random_von_mises_fisher(kappa=kappa, n_samples=n_points)
        sum_points = gs.sum(points, axis=0)
        mean_norm = gs.linalg.norm(sum_points) / n_points
        kappa_estimate = (
            mean_norm * (dim + 1.0 - mean_norm**2) / (1.0 - mean_norm**2)
        )
        kappa_estimate = gs.cast(kappa_estimate, gs.float64)
        p = dim + 1
        n_steps = 100
        for _ in range(n_steps):
            bessel_func_1 = scipy.special.iv(p / 2.0, kappa_estimate)
            bessel_func_2 = scipy.special.iv(p / 2.0 - 1.0, kappa_estimate)
            ratio = bessel_func_1 / bessel_func_2
            denominator = 1.0 - ratio**2 - (p - 1.0) * ratio / kappa_estimate
            mean_norm = gs.cast(mean_norm, gs.float64)
            kappa_estimate = kappa_estimate - (ratio - mean_norm) / denominator
        result = kappa_estimate
        expected = kappa
        self.assertAllClose(result, expected, atol=KAPPA_ESTIMATION_TOL)


class AbstractHypersphereMetric(RiemannianMetricTestCase):
    def test_inner_product(
        self, dim, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        metric = self.metric(dim)
        result = metric.inner_product(
            gs.array(tangent_vec_a), gs.array(tangent_vec_b), gs.array(base_point)
        )
        self.assertAllClose(result, expected)

    def test_dist(self, dim, point_a, point_b, expected):
        metric = self.metric(dim)
        result = metric.dist(gs.array(point_a), gs.array(point_b))
        self.assertAllClose(result, gs.array(expected))

    def test_dist_pairwise(self, dim, point, expected, rtol):
        metric = self.metric(dim)
        result = metric.dist_pairwise(gs.array(point))
        self.assertAllClose(result, gs.array(expected), rtol=rtol)

    def test_diameter(self, dim, points, expected):
        metric = self.metric(dim)
        result = metric.diameter(gs.array(points))
        self.assertAllClose(result, gs.array(expected))

    def test_christoffels_shape(self, dim, point, expected):
        metric = self.metric(dim)
        result = metric.christoffels(point)
        self.assertAllClose(gs.shape(result), expected)

    def test_sectional_curvature(
        self, dim, tangent_vec_a, tangent_vec_b, base_point, expected
    ):
        metric = self.metric(dim)
        result = metric.sectional_curvature(tangent_vec_a, tangent_vec_b, base_point)
        self.assertAllClose(result, expected, atol=1e-2)

    def test_exp_and_dist_and_projection_to_tangent_space(
        self, dim, vector, base_point
    ):
        metric = self.metric(dim)
        tangent_vec = Hypersphere(dim).to_tangent(vector=vector, base_point=base_point)
        exp = metric.exp(tangent_vec=tangent_vec, base_point=base_point)
        result = metric.dist(base_point, exp)
        expected = gs.linalg.norm(tangent_vec) % (2 * gs.pi)
        self.assertAllClose(result, expected)


class TestHypersphereMetric(AbstractHypersphereMetric, metaclass=Parametrizer):
    metric = connection = HypersphereMetric
    skip_test_exp_geodesic_ivp = True
    skip_test_dist_point_to_itself_is_zero = True

    testing_data = HypersphereMetricTestData()
