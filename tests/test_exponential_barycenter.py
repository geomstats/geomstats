"""Unit tests for Exponential Barycenter mean."""

import geomstats.tests
import geomstats.backend as gs
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.exponential_barycenter import ExponentialBarycenter
from geomstats.learning.frechet_mean import FrechetMean


class TestExponentialBarycenter(geomstats.tests.TestCase):

    def setUp(self):
        self.se = SpecialEuclidean(n=3, point_type='matrix')
        self.so = SpecialOrthogonal(n=3, point_type='matrix')
        self.n_samples = 15

    @geomstats.tests.np_only
    def test_estimate_and_belongs_se(self):
        point = self.se.random_uniform(self.n_samples)
        estimator = ExponentialBarycenter(self.se)
        estimator.fit(point)
        barexp = estimator.estimate_
        result = self.se.belongs(barexp)
        expected = True
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_estimate_one_sample_se(self):
        point = self.se.random_uniform(1)
        estimator = ExponentialBarycenter(self.se)
        estimator.fit(point)
        result = estimator.estimate_
        expected = point[0]
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_estimate_and_reach_max_iter_se(self):
        point = self.se.random_uniform(self.n_samples)
        estimator = ExponentialBarycenter(self.se, max_iter=2)
        point = gs.array([point[0], point[0]])
        estimator.fit(point)
        result = estimator.estimate_
        expected = point[0]
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

    @geomstats.tests.np_only
    def test_estimate_one_sample_so(self):
        point = self.so.random_uniform(1)
        estimator = ExponentialBarycenter(self.so)
        estimator.fit(point)
        result = estimator.estimate_
        expected = point[0]
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

    @geomstats.tests.np_only
    def test_coincides_with_frechet_so(self):
        point = self.so.random_uniform(self.n_samples)
        estimator = ExponentialBarycenter(self.so, max_iter=32, epsilon=1e-12)
        estimator.fit(point)
        so_vector = SpecialOrthogonal(3)
        frechet_estimator = FrechetMean(
            so_vector.bi_invariant_metric, max_iter=32, epsilon=1e-10,
            point_type='vector')
        vector_point = so_vector.rotation_vector_from_matrix(point)
        frechet_estimator.fit(vector_point)
        mean = frechet_estimator.estimate_
        expected = so_vector.matrix_from_rotation_vector(mean)[0]
        result = estimator.estimate_
        self.assertAllClose(result, expected)
