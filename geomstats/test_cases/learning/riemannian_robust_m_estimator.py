import pytest

import geomstats.backend as gs
from geomstats.geometry.discrete_curves import SRVMetric
from geomstats.learning.frechet_mean import GradientDescent, variance
from geomstats.test.random import RandomDataGenerator
from geomstats.test.test_case import TestCase
from geomstats.test_cases.learning._base import (
    BaseEstimatorTestCase,
    MeanEstimatorMixinsTestCase,
)
from geomstats.vectorization import repeat_point


class FrechetMeanTestCase(MeanEstimatorMixinsTestCase, BaseEstimatorTestCase):
    @pytest.mark.random
    def test_logs_at_mean(self, atol):
        X = self.data_generator.random_point(n_points=2)

        mean = self.estimator.fit(X).estimate_

        logs = self.estimator.space.metric.log(X, mean)

        result = gs.linalg.norm(logs[1] + logs[0])
        self.assertAllClose(result, gs.array(0.0), atol)

    @pytest.mark.random
    def test_weighted_mean_two_points(self, atol):
        X = self.data_generator.random_point(n_points=2)
        weights = gs.random.rand(2)

        mean = self.estimator.fit(X, weights=weights).estimate_

        space = self.estimator.space
        expected = space.metric.exp(
            weights[1] / gs.sum(weights) * space.metric.log(X[1], X[0]), X[0]
        )
        self.assertAllClose(mean, expected, atol=atol)