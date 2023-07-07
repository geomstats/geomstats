"""Unit tests for Expectation Maximization."""

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.learning.expectation_maximization import (
    GaussianMixtureModel,
    RiemannianEM,
)
from geomstats.learning.frechet_mean import FrechetMean


class TestEM(tests.conftest.TestCase):
    """Class for testing Expectation Maximization."""

    def setup_method(self):
        """Set manifold, data and EM parameters."""
        self.n_samples = 5
        self.dim = 2
        self.space = PoincareBall(dim=self.dim)
        self.initialisation_method = "random"

        cluster_1 = gs.random.uniform(
            low=0.2, high=0.6, size=(self.n_samples, self.dim)
        )
        cluster_2 = gs.random.uniform(
            low=-0.6, high=-0.2, size=(self.n_samples, self.dim)
        )
        cluster_3 = gs.random.uniform(low=-0.3, high=0, size=(self.n_samples, self.dim))
        cluster_3 = cluster_3 * gs.array([-1.0, 1.0])

        self.n_gaussian = 3
        self.data = gs.concatenate((cluster_1, cluster_2, cluster_3), axis=0)

    @tests.conftest.np_and_autograd_only
    def test_fit_init_kmeans(self):
        """Test fitting data into a GMM."""
        gmm_learning = RiemannianEM(
            space=self.space,
            n_gaussians=self.n_gaussian,
            initialisation_method=self.initialisation_method,
        )

        gmm_learning.fit(self.data)
        means = gmm_learning.means_
        variances = gmm_learning.variances_
        coefficients = gmm_learning.mixture_coefficients_

        self.assertTrue((coefficients < 1).all() and (coefficients > 0).all())
        self.assertTrue((variances < 1).all() and (variances > 0).all())
        self.assertTrue(self.space.belongs(means).all())

        gmm_learning = RiemannianEM(
            space=self.space,
            n_gaussians=self.n_gaussian,
            initialisation_method="kmeans",
        )

        gmm_learning.fit(self.data)
        means = gmm_learning.means_
        variances = gmm_learning.variances_
        coefficients = gmm_learning.mixture_coefficients_

        self.assertTrue((coefficients < 1).all() and (coefficients > 0).all())
        self.assertTrue((variances < 1).all() and (variances > 0).all())
        self.assertTrue(self.space.belongs(means).all())

    @tests.conftest.np_and_autograd_only
    def test_fit_init_random(self):
        """Test fitting data into a GMM."""
        gmm_learning = RiemannianEM(
            space=self.space,
            n_gaussians=self.n_gaussian,
            initialisation_method=self.initialisation_method,
        )

        gmm_learning.fit(self.data)
        means = gmm_learning.means_
        variances = gmm_learning.variances_
        coefficients = gmm_learning.mixture_coefficients_

        self.assertTrue((coefficients < 1).all() and (coefficients > 0).all())
        self.assertTrue((variances < 1).all() and (variances > 0).all())
        self.assertTrue(self.space.belongs(means).all())

        gmm_learning = RiemannianEM(
            space=self.space,
            n_gaussians=self.n_gaussian,
            initialisation_method="random",
        )

        gmm_learning.fit(self.data)
        means = gmm_learning.means_
        variances = gmm_learning.variances_
        coefficients = gmm_learning.mixture_coefficients_

        self.assertTrue((coefficients < 1).all() and (coefficients > 0).all())
        self.assertTrue((variances < 1).all() and (variances > 0).all())
        self.assertTrue(self.space.belongs(means).all())

    def test_weighted_frechet_mean(self):
        """Test for weighted mean."""
        data = gs.array([[0.1, 0.2], [0.25, 0.35]])
        weights = gs.array([3.0, 1.0])
        mean_o = FrechetMean(space=self.space)
        mean_o.fit(data, weights=weights)
        result = mean_o.estimate_
        expected = self.space.metric.exp(
            weights[1] / gs.sum(weights) * self.space.metric.log(data[1], data[0]),
            data[0],
        )
        self.assertAllClose(result, expected)

    def test_normalization_factor(self):
        """Test for Gaussian distribution normalization factor."""
        tol = 1e-3

        gmm = GaussianMixtureModel(
            self.space, zeta_lower_bound=5e-2, zeta_upper_bound=2.0, zeta_step=0.001
        )

        self.assertAllClose(gmm.normalization_factor_var[4], 0.00291884, tol)
        self.assertAllClose(gmm.phi_inv_var[3], 0.00562326, tol)

        gmm.variances = variances_test = gs.array([0.8, 1.2])

        norm_factor_test = gmm._compute_normalization_factor()
        norm_factor_verdict = gs.array([0.79577319, 2.3791778])
        self.assertAllClose(norm_factor_test, norm_factor_verdict, tol)

        norm_factor_test2 = self.space.metric.normalization_factor(variances_test)
        self.assertAllClose(norm_factor_test2, norm_factor_verdict, tol)

        (
            norm_factor_test3,
            norm_factor_gradient_test,
        ) = self.space.metric.norm_factor_gradient(variances_test)
        norm_factor_gradient_verdict = gs.array([3.0553115709, 2.53770926])
        self.assertAllClose(norm_factor_test3, norm_factor_verdict, tol)
        self.assertAllClose(
            norm_factor_gradient_test, norm_factor_gradient_verdict, tol
        )

        find_var_test = gmm.compute_variance_from_index(gs.array([0.5, 0.4, 0.3, 0.2]))
        find_var_verdict = gs.array([0.481, 0.434, 0.378, 0.311])
        self.assertAllClose(find_var_test, find_var_verdict, tol)

    @tests.conftest.autograd_only
    def test_fit_init_random_sphere(self):
        """Test fitting data into a GMM."""
        space = Hypersphere(2)
        gmm_learning = RiemannianEM(
            space=space,
            n_gaussians=2,
            initialisation_method=self.initialisation_method,
        )

        means = space.random_uniform(2)
        cluster_1 = space.random_von_mises_fisher(mu=means[0], kappa=20, n_samples=140)
        cluster_2 = space.random_von_mises_fisher(mu=means[1], kappa=20, n_samples=140)

        data = gs.concatenate((cluster_1, cluster_2), axis=0)
        gmm_learning.fit(data)
        means = gmm_learning.means_
        variances = gmm_learning.variances_
        coefficients = gmm_learning.mixture_coefficients_

        self.assertTrue((coefficients < 1).all() and (coefficients > 0).all())
        self.assertTrue((variances < 1).all() and (variances > 0).all())
        self.assertTrue(space.belongs(means).all())
