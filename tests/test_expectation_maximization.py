"""Unit tests for Expectation Maximization."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.learning.expectation_maximization import RiemannianEM
from geomstats.learning.frechet_mean import FrechetMean

TOLERANCE = 1e-3
ZETA_LOWER_BOUND = 5e-2
ZETA_UPPER_BOUND = 2.
ZETA_STEP = 0.001


class TestEM(geomstats.tests.TestCase):
    """Class for testing Expectation Maximization."""
    @geomstats.tests.np_and_pytorch_only
    def setUp(self):
        """Set manifold, data and EM parameters."""
        self.n_samples = 5
        self.dim = 2
        self.space = PoincareBall(dim=self.dim)
        self.metric = self.space.metric
        self.initialisation_method = 'random'
        self.mean_method = 'frechet-poincare-ball'

        cluster_1 = gs.random.uniform(
            low=0.2, high=0.6, size=(self.n_samples, self.dim))
        cluster_2 = gs.random.uniform(
            low=-0.6, high=-0.2, size=(self.n_samples, self.dim))
        cluster_3 = gs.random.uniform(
            low=-0.3, high=0, size=(self.n_samples, self.dim))
        cluster_3[:, 0] = -cluster_3[:, 0]

        self.n_gaussian = 3
        self.data = gs.concatenate((cluster_1, cluster_2, cluster_3), axis=0)

    @geomstats.tests.np_only
    def test_fit(self):
        """Test fitting data into a GMM."""
        gmm_learning = RiemannianEM(
            metric=self.metric,
            n_gaussians=self.n_gaussian,
            initialisation_method=self.initialisation_method,
            mean_method=self.mean_method)

        means, variances, coefficients = gmm_learning.fit(self.data)

        self.assertTrue((coefficients < 1).all() and (coefficients > 0).all())
        self.assertTrue((variances < 1).all() and (variances > 0).all())
        self.assertTrue(self.space.belongs(means).all())

    @geomstats.tests.np_only
    def test_weighted_frechet_mean(self):
        """Test for weighted mean."""
        data = gs.array([[0.1, 0.2],
                         [0.25, 0.35],
                        [-0.1, -0.2],
                        [-0.4, 0.3]])
        weights = gs.repeat([0.5], data.shape[0])
        mean_o = FrechetMean(
            metric=self.metric,
            point_type='vector')
        mean_o.fit(data, weights)
        mean = mean_o.estimate_
        mean_verdict = [-0.03857, 0.15922]
        self.assertAllClose(mean, mean_verdict, TOLERANCE)

    @geomstats.tests.np_and_pytorch_only
    def test_normalization_factor(self):
        """Test for Gaussian distribution normalization factor."""
        variances_range,\
            normalization_factor_var,\
            phi_inv_var = \
            self.metric.normalization_factor_init(gs.arange(
                ZETA_LOWER_BOUND, ZETA_UPPER_BOUND, ZETA_STEP))
        self.assertAllClose(
            normalization_factor_var[4], 0.00291884, TOLERANCE)
        self.assertAllClose(phi_inv_var[3], 0.00562326, TOLERANCE)

        variances_test = gs.array([0.8, 1.2])
        norm_factor_test = self.metric.find_normalization_factor(
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

        find_var_test = self.metric.find_variance_from_index(
            gs.array([0.5, 0.4, 0.3, 0.2]), variances_range, phi_inv_var)
        find_var_verdict = gs.array([0.481, 0.434, 0.378, 0.311])
        self.assertAllClose(find_var_test, find_var_verdict, TOLERANCE)
