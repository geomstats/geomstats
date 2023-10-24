import geomstats.backend as gs
from geomstats.test.data import TestData

from ._base import BaseEstimatorTestData


class RiemannianEMTestData(BaseEstimatorTestData):
    fail_for_autodiff_exceptions = False

    MIN_RANDOM = 5
    MAX_RANDOM = 8

    def estimate_belongs_test_data(self):
        return self.generate_random_data()

    def fit_coefficients_and_variances_bounds_test_data(self):
        return self.generate_random_data()


class GaussianMixtureModelTestData(TestData):
    def normalization_factor_init_test_data(self):
        data = [
            dict(
                expected_normalization_factor_var=0.00291884,
                expected_phi_inv_var=0.00562326,
                atol=1e-6,
            )
        ]
        return self.generate_tests(data)

    def normalization_factor_test_data(self):
        data = [
            dict(
                expected_norm_factor=gs.array([0.79577319, 2.3791778]),
                atol=1e-3,
            ),
        ]
        return self.generate_tests(data)

    def metric_normalization_factor_test_data(self):
        data = [
            dict(
                expected_norm_factor=gs.array([0.79577319, 2.3791778]),
                atol=1e-3,
            )
        ]
        return self.generate_tests(data)

    def metric_norm_factor_gradient_test_data(self):
        data = [
            dict(
                expected_norm_factor=gs.array([0.79577319, 2.3791778]),
                expected_norm_factor_gradient=gs.array([3.0553115709, 2.53770926]),
                atol=1e-3,
            )
        ]
        return self.generate_tests(data)

    def compute_variance_from_index_test_data(self):
        data = [
            dict(
                weighted_distances=gs.array([0.5, 0.4, 0.3, 0.2]),
                expected_var=gs.array([0.481, 0.434, 0.378, 0.311]),
            )
        ]
        return self.generate_tests(data)
