import pytest

from geomstats.test_cases.learning._base import BaseEstimatorTestCase


class IncrementalFrechetMeanTestCase(BaseEstimatorTestCase):
    def test_fit(self, X, expected, atol):
        estimate = self.estimator.fit(X).estimate_
        self.assertAllClose(estimate, expected, atol=atol)

    @pytest.mark.random
    def test_estimate_belongs(self, n_points, atol):
        X = self.data_generator.random_point(n_points=n_points)
        mean = self.estimator.fit(X).estimate_
        belongs = self.estimator.space.belongs(mean, atol=atol)
        self.assertTrue(belongs)
