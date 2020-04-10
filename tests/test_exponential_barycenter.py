"""Unit tests for Exponential Barycenter mean."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.exponential_barycenter import ExponentialBarycenter
from geomstats.learning.frechet_mean import FrechetMean


class TestExponentialBarycenter(geomstats.tests.TestCase):

    def setUp(self):
        self.se_mat = SpecialEuclidean(n=3, point_type='matrix')
        self.so_vec = SpecialOrthogonal(n=3, point_type='vector')
        self.so = SpecialOrthogonal(n=3, point_type='matrix')
        self.n_samples = 3

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
        point = self.se_mat.random_uniform(1)
        estimator = ExponentialBarycenter(self.se_mat)
        estimator.fit(point)
        result = estimator.estimate_
        expected = point[0]
        self.assertAllClose(result, expected)

        point = self.so_vec.random_uniform(1)
        estimator = ExponentialBarycenter(self.so_vec)
        estimator.fit(point)
        result = estimator.estimate_
        expected = point[0]
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_estimate_and_reach_max_iter_se(self):
        point = self.se_mat.random_uniform(self.n_samples)
        estimator = ExponentialBarycenter(self.se_mat, max_iter=2)
        point = gs.array([point[0], point[0]])
        estimator.fit(point)
        result = estimator.estimate_
        expected = point[0]
        self.assertAllClose(result, expected)

        point = self.so_vec.random_uniform(self.n_samples)
        estimator = ExponentialBarycenter(self.so_vec, max_iter=2)
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
        expected = point[0]
        self.assertAllClose(result, expected)

        point = self.so_vec.random_uniform(1)
        estimator = ExponentialBarycenter(self.so_vec)
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

        point = self.so_vec.random_uniform(self.n_samples)
        estimator = ExponentialBarycenter(self.so_vec, max_iter=2)
        estimator.fit(point)
        barexp = estimator.estimate_
        result = self.so_vec.belongs(barexp)
        expected = True
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_coincides_with_frechet_so(self):
        point = self.so.random_uniform(self.n_samples)
        estimator = ExponentialBarycenter(self.so, max_iter=32, epsilon=1e-12)
        estimator.fit(point)
        so_vector = SpecialOrthogonal(3, point_type='vector')
        frechet_estimator = FrechetMean(
            so_vector.bi_invariant_metric, max_iter=32, epsilon=1e-10,
            point_type='vector')
        vector_point = so_vector.rotation_vector_from_matrix(point)
        frechet_estimator.fit(vector_point)
        mean = frechet_estimator.estimate_
        expected = so_vector.matrix_from_rotation_vector(mean)[0]
        result = estimator.estimate_
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
        se_mat = SpecialEuclidean(n=3, point_type='vector')
        translations = se_mat.translations
        point = translations.random_uniform(self.n_samples)
        estimator = ExponentialBarycenter(translations)
        estimator.fit(point)
        result = estimator.estimate_
        expected = gs.mean(point, axis=0)
        self.assertAllClose(result, expected)
