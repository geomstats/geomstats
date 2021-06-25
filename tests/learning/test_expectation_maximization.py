"""Unit tests for Expectation Maximization."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.learning.expectation_maximization import \
    find_normalization_factor, find_variance_from_index, RiemannianEM
from geomstats.learning.frechet_mean import FrechetMean

TOLERANCE = 1e-3
ZETA_LOWER_BOUND = 5e-2
ZETA_UPPER_BOUND = 2.
ZETA_STEP = 0.001


class TestEM(geomstats.tests.TestCase):
    """Class for testing Expectation Maximization."""

    def setUp(self):
        """Set manifold, data and EM parameters."""
        self.n_samples = 5
        self.dim = 2
        self.space = PoincareBall(dim=self.dim)
        self.metric = self.space.metric
        self.initialisation_method = 'random'
        self.mean_method = 'batch'

        cluster_1 = gs.random.uniform(
            low=0.2, high=0.6, size=(self.n_samples, self.dim))
        cluster_2 = gs.random.uniform(
            low=-0.6, high=-0.2, size=(self.n_samples, self.dim))
        cluster_3 = gs.random.uniform(
            low=-0.3, high=0, size=(self.n_samples, self.dim))
        cluster_3 = cluster_3 * gs.array([-1., 1.])

        self.n_gaussian = 3
        self.data = gs.concatenate((cluster_1, cluster_2, cluster_3), axis=0)

    @geomstats.tests.np_only
    def test_fit_init_kmeans(self):
        """Test fitting data into a GMM."""
        gmm_learning = RiemannianEM(
            metric=self.metric,
            n_gaussians=self.n_gaussian,
            initialisation_method=self.initialisation_method)

        means, variances, coefficients = gmm_learning.fit(self.data)

        self.assertTrue((coefficients < 1).all() and (coefficients > 0).all())
        self.assertTrue((variances < 1).all() and (variances > 0).all())
        self.assertTrue(self.space.belongs(means).all())

        gmm_learning = RiemannianEM(
            metric=self.metric,
            n_gaussians=self.n_gaussian,
            initialisation_method='kmeans')

        means, variances, coefficients = gmm_learning.fit(self.data)

        self.assertTrue((coefficients < 1).all() and (coefficients > 0).all())
        self.assertTrue((variances < 1).all() and (variances > 0).all())
        self.assertTrue(self.space.belongs(means).all())

    @geomstats.tests.np_only
    def test_fit_init_random(self):
        """Test fitting data into a GMM."""
        gmm_learning = RiemannianEM(
            metric=self.metric,
            n_gaussians=self.n_gaussian,
            initialisation_method=self.initialisation_method)

        means, variances, coefficients = gmm_learning.fit(self.data)

        self.assertTrue((coefficients < 1).all() and (coefficients > 0).all())
        self.assertTrue((variances < 1).all() and (variances > 0).all())
        self.assertTrue(self.space.belongs(means).all())

        gmm_learning = RiemannianEM(
            metric=self.metric,
            n_gaussians=self.n_gaussian,
            initialisation_method='random')

        means, variances, coefficients = gmm_learning.fit(self.data)

        self.assertTrue((coefficients < 1).all() and (coefficients > 0).all())
        self.assertTrue((variances < 1).all() and (variances > 0).all())
        self.assertTrue(self.space.belongs(means).all())

    def test_weighted_frechet_mean(self):
        """Test for weighted mean."""
        data = gs.array([[0.1, 0.2],
                         [0.25, 0.35]])
        weights = gs.array([3., 1.])
        mean_o = FrechetMean(
            metric=self.metric, point_type='vector', lr=1.)
        mean_o.fit(data, weights=weights)
        result = mean_o.estimate_
        expected = self.metric.exp(
            weights[1] / gs.sum(weights) * self.metric.log(data[1], data[0]),
            data[0])
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_normalization_factor(self):
        """Test for Gaussian distribution normalization factor."""
        gmm = RiemannianEM(self.metric)
        variances_range, normalization_factor_var, phi_inv_var = \
            gmm.normalization_factor_init(
                gs.arange(ZETA_LOWER_BOUND, ZETA_UPPER_BOUND, ZETA_STEP))
        self.assertAllClose(
            normalization_factor_var[4], 0.00291884, TOLERANCE)
        self.assertAllClose(phi_inv_var[3], 0.00562326, TOLERANCE)

        variances_test = gs.array([0.8, 1.2])
        norm_factor_test = find_normalization_factor(
            variances_test, variances_range, normalization_factor_var)
        norm_factor_verdict = gs.array([0.79577319, 2.3791778])
        self.assertAllClose(norm_factor_test, norm_factor_verdict, TOLERANCE)

        norm_factor_test2 = self.metric.normalization_factor(variances_test)
        self.assertAllClose(norm_factor_test2, norm_factor_verdict, TOLERANCE)

        norm_factor_test3, norm_factor_gradient_test = \
            self.metric.norm_factor_gradient(variances_test)
        norm_factor_gradient_verdict = gs.array([3.0553115709, 2.53770926])
        self.assertAllClose(norm_factor_test3, norm_factor_verdict, TOLERANCE)
        self.assertAllClose(
            norm_factor_gradient_test, norm_factor_gradient_verdict, TOLERANCE)

        find_var_test = find_variance_from_index(
            gs.array([0.5, 0.4, 0.3, 0.2]), variances_range, phi_inv_var)
        find_var_verdict = gs.array([0.481, 0.434, 0.378, 0.311])
        self.assertAllClose(find_var_test, find_var_verdict, TOLERANCE)

    @geomstats.tests.np_only
    def test_fit_init_random_sphere(self):
        """Test fitting data into a GMM."""
        space = Hypersphere(2)
        gmm_learning = RiemannianEM(
            metric=space.metric,
            n_gaussians=2,
            initialisation_method=self.initialisation_method)

        means = space.random_uniform(2)
        cluster_1 = space.random_von_mises_fisher(
            mu=means[0], kappa=20, n_samples=140)
        cluster_2 = space.random_von_mises_fisher(
            mu=means[1], kappa=20, n_samples=140)

        data = gs.concatenate((cluster_1, cluster_2), axis=0)
        means, variances, coefficients = gmm_learning.fit(data)

        self.assertTrue((coefficients < 1).all() and (coefficients > 0).all())
        self.assertTrue((variances < 1).all() and (variances > 0).all())
        self.assertTrue(space.belongs(means).all())
