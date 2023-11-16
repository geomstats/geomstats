import pytest

import geomstats.backend as gs
from geomstats.geometry.discrete_curves import SRVTranslationMetric
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


class ElasticMeanTestCase(FrechetMeanTestCase):
    @pytest.mark.random
    def test_logs_at_mean(self, atol):
        space = self.estimator.space
        if not isinstance(space.metric, SRVTranslationMetric):
            return

        X = self.data_generator.random_point(n_points=2)

        mean = self.estimator.fit(X).estimate_

        logs = space.metric.log(X, mean)
        logs_srv = space.metric.diffeo.tangent_diffeomorphism(logs, base_point=mean)
        result = gs.linalg.norm(logs_srv[1] + logs_srv[0])

        self.assertAllClose(result, gs.array(0.0), atol)


class CircularMeanTestCase(FrechetMeanTestCase):
    @pytest.mark.random
    def test_against_optimization(self, n_points, atol):
        space = self.estimator.space
        X = self.data_generator.random_point(n_points)

        mean = self.estimator.fit(X).estimate_

        mean_gd = GradientDescent().minimize(space, points=X, weights=None)

        sum_sd_mean = gs.sum(space.metric.squared_dist(X, mean))
        sum_sd_mean_gd = gs.sum(space.metric.squared_dist(X, mean_gd))

        msg = f"circular mean: {mean}, {sum_sd_mean}\ngd: {mean_gd}, {sum_sd_mean_gd}"
        self.assertTrue(sum_sd_mean < sum_sd_mean_gd + atol, msg)


class VarianceTestCase(TestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)

    def test_variance(self, points, base_point, expected, atol, weights=None):
        res = variance(self.space, points, base_point, weights=weights)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_variance_repeated_is_zero(self, n_samples, atol):
        base_point = point = self.data_generator.random_point(n_points=1)
        points = repeat_point(point, n_samples)

        self.test_variance(points, base_point, 0.0, atol)


class BatchGradientDescentTestCase(TestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)

    @pytest.mark.vec
    def test_against_default(self, n_points, n_reps, atol):
        points = self.data_generator.random_point(n_points)

        res_single = self.optimizer.minimize(self.space, points)

        rep_points = repeat_point(points, n_reps)
        rep_points = gs.moveaxis(rep_points, 0, 1)

        res = self.batch_optimizer.minimize(self.space, rep_points)

        self.assertAllClose(res, repeat_point(res_single, n_reps), atol)
