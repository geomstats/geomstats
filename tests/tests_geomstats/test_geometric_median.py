"""Methods for testing the geometric median."""

import geomstats.backend as gs
from tests.conftest import Parametrizer, TestCase
from tests.data.geometric_median_data import GeometricMedianTestData


class TestGeometricMedian(TestCase, metaclass=Parametrizer):
    testing_data = GeometricMedianTestData()

    def test_fit(self, estimator, X, expected):
        estimator.fit(X)
        self.assertAllClose(
            estimator.estimate_,
            expected,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_fit_sanity(self, estimator):
        # Test estimate belongs to space,
        # and weights=None is equivalent to uniform weights.
        n_samples = 5
        space = estimator.space
        X = space.random_point(n_samples)

        med = estimator.fit(X).estimate_
        self.assertTrue(space.belongs(med))

        weights = gs.ones(n_samples)
        wmed = estimator.fit(X, None, weights).estimate_
        self.assertAllClose(med, wmed)
