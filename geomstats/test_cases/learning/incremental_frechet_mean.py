from geomstats.test_cases.learning._base import (
    BaseEstimatorTestCase,
    MeanEstimatorMixinsTestCase,
)


class IncrementalFrechetMeanTestCase(
    MeanEstimatorMixinsTestCase, BaseEstimatorTestCase
):
    def test_fit(self, X, expected, atol):
        estimate = self.estimator.fit(X).estimate_
        self.assertAllClose(estimate, expected, atol=atol)
