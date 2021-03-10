"""Unit tests for Exponential Barycenter mean."""

import logging

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.exponential_barycenter import ExponentialBarycenter
from geomstats.learning.frechet_mean import FrechetMean


class TestExponentialBarycenter(geomstats.tests.TestCase):

    def setUp(self):
        logger = logging.getLogger()
        logger.disabled = True
        self.se_mat = SpecialEuclidean(n=3)
        self.so_vec = SpecialOrthogonal(n=3, point_type='vector')
        self.so = SpecialOrthogonal(n=3)
        self.n_samples = 4

    @geomstats.tests.np_only
    def test_estimate_and_belongs_se(self):
        point = self.se_mat.random_uniform(self.n_samples)
        estimator = ExponentialBarycenter(self.se_mat)
        estimator.fit(point)
        barexp = estimator.estimate_
        result = self.se_mat.belongs(barexp)
        expected = True
        self.assertAllClose(result, expected)

        point = self.so_vec.random_uniform(self.n_samples)
        estimator = ExponentialBarycenter(self.so_vec)
        estimator.fit(point)
        barexp = estimator.estimate_
        result = self.so_vec.belongs(barexp)
        expected = True
        self.assertAllClose(result, expected)

    def test_estimate_one_sample_se(self):
        point = self.se_mat.random_uniform()
        estimator = ExponentialBarycenter(self.se_mat)
        estimator.fit(point)
        result = estimator.estimate_
        expected = point
        self.assertAllClose(result, expected)

        point = self.so_vec.random_uniform(1)
        estimator = ExponentialBarycenter(self.so_vec)
        estimator.fit(point)
        result = estimator.estimate_
        expected = point
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_estimate_and_reach_max_iter_se(self):
        point = self.se_mat.random_uniform(1)
        estimator = ExponentialBarycenter(self.se_mat, max_iter=2)
        points = gs.array([point, point])
        estimator.fit(points)
        result = estimator.estimate_
        expected = point
        self.assertAllClose(result, expected)

        point = self.so_vec.random_uniform(1)
        estimator = ExponentialBarycenter(self.so_vec, max_iter=2)
        points = gs.array([point, point])
        estimator.fit(points)
        result = estimator.estimate_
        expected = point
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_estimate_so_matrix(self):
        points = self.so.random_uniform(2)

        mean_vec = ExponentialBarycenter(group=self.so)
        mean_vec.fit(points)

        logs = self.so.log(points, mean_vec.estimate_)
        result = gs.sum(logs, axis=0)
        expected = gs.zeros_like(points[0])
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_estimate_and_belongs_so(self):
        point = self.so.random_uniform(self.n_samples)
        estimator = ExponentialBarycenter(self.so)
        estimator.fit(point)
        barexp = estimator.estimate_
        result = self.so.belongs(barexp)
        expected = True
        self.assertAllClose(result, expected)

        point = self.so_vec.random_uniform(self.n_samples)
        estimator = ExponentialBarycenter(self.so_vec)
        estimator.fit(point)
        barexp = estimator.estimate_
        result = self.so_vec.belongs(barexp)
        expected = True
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_estimate_one_sample_so(self):
        point = self.so.random_uniform(1)
        estimator = ExponentialBarycenter(self.so)
        estimator.fit(point)
        result = estimator.estimate_
        expected = point
        self.assertAllClose(result, expected)

        point = self.so_vec.random_uniform(1)
        estimator = ExponentialBarycenter(self.so_vec)
        estimator.fit(point)
        result = estimator.estimate_
        expected = point
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_estimate_and_reach_max_iter_so(self):
        point = self.so.random_uniform(self.n_samples)
        estimator = ExponentialBarycenter(self.so, max_iter=2)
        estimator.fit(point)
        barexp = estimator.estimate_
        result = self.so.belongs(barexp)
        expected = True
        self.assertAllClose(result, expected)

        point = self.so_vec.random_uniform(self.n_samples)
        estimator = ExponentialBarycenter(self.so_vec, max_iter=2)
        estimator.fit(point)
        barexp = estimator.estimate_
        result = self.so_vec.belongs(barexp)
        expected = True
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_coincides_with_frechet_so(self):
        gs.random.seed(0)
        point = self.so.random_uniform(self.n_samples)
        estimator = ExponentialBarycenter(self.so, max_iter=40, epsilon=1e-10)
        estimator.fit(point)
        result = estimator.estimate_
        frechet_estimator = FrechetMean(
            self.so.bi_invariant_metric, max_iter=40, epsilon=1e-10)
        frechet_estimator.fit(point)
        expected = frechet_estimator.estimate_
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_estimate_weights(self):
        point = self.so.random_uniform(self.n_samples)
        estimator = ExponentialBarycenter(self.so, verbose=True)
        weights = gs.arange(self.n_samples)
        estimator.fit(point, weights=weights)
        barexp = estimator.estimate_
        result = self.so.belongs(barexp)
        expected = True
        self.assertAllClose(result, expected)

        point = self.so_vec.random_uniform(self.n_samples)
        estimator = ExponentialBarycenter(self.so_vec)
        estimator.fit(point, weights=weights)
        barexp = estimator.estimate_
        result = self.so_vec.belongs(barexp)
        expected = True
        self.assertAllClose(result, expected)

    def test_linear_mean(self):
        euclidean = Euclidean(3)
        point = euclidean.random_uniform(self.n_samples)

        estimator = ExponentialBarycenter(euclidean)

        estimator.fit(point)
        result = estimator.estimate_

        expected = gs.mean(point, axis=0)

        self.assertAllClose(result, expected)
