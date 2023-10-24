import pytest

import geomstats.backend as gs
from geomstats.test.test_case import TestCase
from geomstats.test_cases.learning._base import BaseEstimatorTestCase


class RiemannianEMTestCase(BaseEstimatorTestCase):
    @pytest.mark.random
    def test_estimate_belongs(self, n_samples, atol):
        X = self.data_generator.random_point(n_points=n_samples)
        means = self.estimator.fit(X).means_
        belongs = self.estimator.space.belongs(means, atol=atol)
        self.assertAllEqual(belongs, gs.ones(self.estimator.n_gaussians, dtype=bool))

    @pytest.mark.random
    def test_fit_coefficients_and_variances_bounds(self, n_samples, atol):
        X = self.data_generator.random_point(n_points=n_samples)
        self.estimator.fit(X)

        coefficients = self.estimator.mixture_coefficients_
        variances = self.estimator.variances_

        self.assertTrue((coefficients < 1).all() and (coefficients > 0).all())
        self.assertTrue((variances < 1).all() and (variances > 0).all())


class GaussianMixtureModelTestCase(TestCase):
    def test_normalization_factor_init(
        self, expected_normalization_factor_var, expected_phi_inv_var, atol
    ):
        self.assertAllClose(
            self.model.normalization_factor_var[4],
            expected_normalization_factor_var,
            atol=atol,
        )
        self.assertAllClose(self.model.phi_inv_var[3], expected_phi_inv_var, atol=atol)

    def test_normalization_factor(self, expected_norm_factor, atol):
        norm_factor = self.model._compute_normalization_factor()
        self.assertAllClose(norm_factor, expected_norm_factor, atol=atol)

    def test_metric_normalization_factor(self, expected_norm_factor, atol):
        norm_factor = self.space.metric.normalization_factor(self.model.variances)
        self.assertAllClose(norm_factor, expected_norm_factor, atol=atol)

    def test_metric_norm_factor_gradient(
        self, expected_norm_factor, expected_norm_factor_gradient, atol
    ):
        (
            norm_factor,
            norm_factor_gradient,
        ) = self.model.space.metric.norm_factor_gradient(self.model.variances)

        self.assertAllClose(norm_factor, expected_norm_factor, atol=atol)
        self.assertAllClose(
            norm_factor_gradient, expected_norm_factor_gradient, atol=atol
        )

    def test_compute_variance_from_index(self, weighted_distances, expected_var, atol):
        var = self.model.compute_variance_from_index(weighted_distances)
        self.assertAllClose(var, expected_var, atol=atol)
