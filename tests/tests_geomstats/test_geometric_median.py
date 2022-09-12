"""Methods for testing the geometric median."""


from tests.conftest import Parametrizer, TestCase
from tests.data.geometric_median_data import GeometricMedianTestData


class TestGeometricMedian(TestCase, metaclass=Parametrizer):
    testing_data = GeometricMedianTestData()

    def test_fit(self, estimator, X, expected):
        estimator.fit(X)
        self.assertAllClose(estimator.estimate_, expected)

    def test_fit_sanity(self, estimator, space):
        """Test estimate belongs to space."""
        n_samples = 5

        X = space.random_point(n_samples)
        estimator.fit(X)

        self.assertTrue(space.belongs(estimator.estimate_))
