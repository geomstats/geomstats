import pytest

import geomstats.backend as gs
from geomstats.geometry.discrete_curves import SRVMetric
from geomstats.learning.frechet_mean import GradientDescent
from geomstats.test.random import RandomDataGenerator
from geomstats.test.test_case import TestCase
from geomstats.vectorization import repeat_point

# TODO: inherit from BaseEstimatorTestCase


class FrechetMeanTestCase(TestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.estimator.space)

    @pytest.mark.random
    def test_one_point(self, atol):
        X = gs.expand_dims(self.data_generator.random_point(n_points=1), axis=0)

        mean = self.estimator.fit(X).estimate_
        self.assertAllClose(mean, X[0], atol)

    @pytest.mark.random
    def test_n_times_same_point(self, n_reps, atol):
        X = repeat_point(self.data_generator.random_point(n_points=1), n_reps)

        mean = self.estimator.fit(X).estimate_
        self.assertAllClose(mean, X[0], atol)

    @pytest.mark.random
    def test_estimate_belongs(self, n_points, atol):
        X = self.data_generator.random_point(n_points=n_points)
        mean = self.estimator.fit(X).estimate_
        belongs = self.estimator.space.belongs(mean, atol=atol)
        self.assertTrue(belongs)

    @pytest.mark.random
    def test_logs_at_mean(self, atol):
        X = self.data_generator.random_point(n_points=2)

        mean = self.estimator.fit(X).estimate_

        logs = self.estimator.space.metric.log(X, mean)

        result = gs.linalg.norm(logs[1] + logs[0])
        self.assertAllClose(result, gs.array(0.0), atol)


class ElasticMeanTestCase(FrechetMeanTestCase):
    @pytest.mark.random
    def test_logs_at_mean(self, atol):
        space = self.estimator.space
        if not isinstance(space.metric, SRVMetric):
            return

        X = self.data_generator.random_point(n_points=2)

        mean = self.estimator.fit(X).estimate_

        logs = space.metric.log(X, mean)
        logs_srv = space.metric.tangent_diffeomorphism(logs, base_point=mean)
        result = gs.linalg.norm(logs_srv[1] + logs_srv[0])

        self.assertAllClose(result, gs.array(0.0), atol)


class CircularMeanTestCase(FrechetMeanTestCase):
    @pytest.mark.random
    def test_against_optimization(self, n_points, atol):
        X = self.data_generator.random_point(n_points)
        space = self.estimator.space

        mean = self.estimator.fit(X).estimate_

        optimizer = GradientDescent()
        mean_gd = optimizer.minimize(space, points=X, weights=None)

        sum_sd_mean = gs.sum(space.metric.dist(X, mean) ** 2)
        sum_sd_mean_gd = gs.sum(space.metric.dist(X, mean_gd) ** 2)

        msg = f"circular mean: {mean}, {sum_sd_mean}\ngd: {mean_gd}, {sum_sd_mean_gd}"
        self.assertTrue(sum_sd_mean < sum_sd_mean_gd + atol, msg)


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
