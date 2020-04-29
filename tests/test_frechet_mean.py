"""Unit tests for Frechet mean."""

import math

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.matrices import MatricesMetric
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.spd_matrices import SPDMatrices, SPDMetricAffine
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.frechet_mean import variance


class TestFrechetMean(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(123)
        self.sphere = Hypersphere(dim=4)
        self.hyperbolic = Hyperboloid(dim=3)
        self.euclidean = Euclidean(dim=2)
        self.minkowski = Minkowski(dim=2)
        self.so3 = SpecialOrthogonal(n=3, point_type='vector')
        self.so_matrix = SpecialOrthogonal(n=3)

    @geomstats.tests.np_only
    def test_logs_at_mean_default_gradient_descent_sphere(self):
        n_tests = 100
        estimator = FrechetMean(metric=self.sphere.metric, method='default')

        result = gs.zeros(n_tests)
        for i in range(n_tests):
            # take 2 random points, compute their mean, and verify that
            # log of each at the mean is opposite
            points = self.sphere.random_uniform(n_samples=2)
            estimator.fit(points)
            mean = estimator.estimate_

            logs = self.sphere.metric.log(point=points, base_point=mean)
            result[i] = gs.linalg.norm(logs[1, :] + logs[0, :])

        expected = gs.zeros(n_tests)
        self.assertAllClose(expected, result, rtol=1e-10, atol=1e-10)

    @geomstats.tests.np_only
    def test_logs_at_mean_adaptive_gradient_descent_sphere(self):
        n_tests = 100
        estimator = FrechetMean(metric=self.sphere.metric, method='adaptive')

        result = gs.zeros(n_tests)
        for i in range(n_tests):
            # take 2 random points, compute their mean, and verify that
            # log of each at the mean is opposite
            points = self.sphere.random_uniform(n_samples=2)
            estimator.fit(points)
            mean = estimator.estimate_

            logs = self.sphere.metric.log(point=points, base_point=mean)
            result[i] = gs.linalg.norm(logs[1, :] + logs[0, :])

        expected = gs.zeros(n_tests)
        self.assertAllClose(expected, result, rtol=1e-10, atol=1e-10)

    @geomstats.tests.np_and_pytorch_only
    def test_estimate_shape_default_gradient_descent_sphere(self):
        dim = 5
        point_a = gs.array([1., 0., 0., 0., 0.])
        point_b = gs.array([0., 1., 0., 0., 0.])
        points = gs.array([point_a, point_b])

        mean = FrechetMean(metric=self.sphere.metric, method='default')
        mean.fit(points)
        result = mean.estimate_

        self.assertAllClose(gs.shape(result), (dim,))

    @geomstats.tests.np_and_pytorch_only
    def test_estimate_shape_adaptive_gradient_descent_sphere(self):
        dim = 5
        point_a = gs.array([1., 0., 0., 0., 0.])
        point_b = gs.array([0., 1., 0., 0., 0.])
        points = gs.array([point_a, point_b])

        mean = FrechetMean(metric=self.sphere.metric, method='adaptive')
        mean.fit(points)
        result = mean.estimate_

        self.assertAllClose(gs.shape(result), (dim,))

    @geomstats.tests.np_and_pytorch_only
    def test_estimate_and_belongs_default_gradient_descent_sphere(self):
        point_a = gs.array([1., 0., 0., 0., 0.])
        point_b = gs.array([0., 1., 0., 0., 0.])
        points = gs.array([point_a, point_b])

        mean = FrechetMean(metric=self.sphere.metric, method='default')
        mean.fit(points)

        result = self.sphere.belongs(mean.estimate_)
        expected = True
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_estimate_default_gradient_descent_so3(self):
        points = self.so3.random_uniform(2)

        mean_vec = FrechetMean(
            metric=self.so3.bi_invariant_metric, method='default')
        mean_vec.fit(points)

        logs = self.so3.bi_invariant_metric.log(points, mean_vec.estimate_)
        result = gs.sum(logs, axis=0)
        expected = gs.zeros_like(points[0])
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_estimate_and_belongs_default_gradient_descent_so3(self):
        point = self.so3.random_uniform(10)

        mean_vec = FrechetMean(
            metric=self.so3.bi_invariant_metric, method='default')
        mean_vec.fit(point)

        result = self.so3.belongs(mean_vec.estimate_)
        expected = True
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_estimate_default_gradient_descent_so_matrix(self):
        points = self.so_matrix.random_uniform(2)

        mean_vec = FrechetMean(
            metric=self.so_matrix.bi_invariant_metric, method='default')
        mean_vec.fit(points)

        logs = self.so_matrix.bi_invariant_metric.log(
            points, mean_vec.estimate_)
        result = gs.sum(logs, axis=0)
        expected = gs.zeros_like(points[0])
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_estimate_and_belongs_default_gradient_descent_so_matrix(self):
        point = self.so_matrix.random_uniform(10)

        mean = FrechetMean(
            metric=self.so_matrix.bi_invariant_metric, method='default')
        mean.fit(point)

        result = self.so_matrix.belongs(mean.estimate_)
        expected = True
        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_estimate_and_coincide_default_so_vec_and_mat(self):
        point = self.so_matrix.random_uniform(10)

        mean = FrechetMean(
            metric=self.so_matrix.bi_invariant_metric, method='default')
        mean.fit(point)
        expected = mean.estimate_

        mean_vec = FrechetMean(
            metric=self.so3.bi_invariant_metric, method='default')
        point_vec = self.so3.rotation_vector_from_matrix(point)
        mean_vec.fit(point_vec)
        result_vec = mean_vec.estimate_
        result = self.so3.matrix_from_rotation_vector(result_vec)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_estimate_and_belongs_adaptive_gradient_descent_sphere(self):
        point_a = gs.array([1., 0., 0., 0., 0.])
        point_b = gs.array([0., 1., 0., 0., 0.])
        points = gs.array([point_a, point_b])

        mean = FrechetMean(metric=self.sphere.metric, method='adaptive')
        mean.fit(points)

        result = self.sphere.belongs(mean.estimate_)
        expected = True
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_variance_sphere(self):
        point = gs.array([0., 0., 0., 0., 1.])
        points = gs.array([point, point])

        result = variance(
            points, base_point=point, metric=self.sphere.metric)
        expected = gs.array(0.)

        self.assertAllClose(expected, result)

    @geomstats.tests.np_and_pytorch_only
    def test_estimate_default_gradient_descent_sphere(self):
        point = gs.array([0., 0., 0., 0., 1.])
        points = gs.array([point, point])

        mean = FrechetMean(metric=self.sphere.metric, method='default')
        mean.fit(X=points)

        result = mean.estimate_
        expected = point

        self.assertAllClose(expected, result)

    @geomstats.tests.np_and_pytorch_only
    def test_estimate_adaptive_gradient_descent_sphere(self):
        point = gs.array([0., 0., 0., 0., 1.])
        points = gs.array([point, point])

        mean = FrechetMean(metric=self.sphere.metric, method='adaptive')
        mean.fit(X=points)

        result = mean.estimate_
        expected = point

        self.assertAllClose(expected, result)

    @geomstats.tests.np_and_pytorch_only
    def test_estimate_spd(self):
        point = SPDMatrices(3).random_uniform()
        points = gs.array([point, point])
        mean = FrechetMean(metric=SPDMetricAffine(3), point_type='matrix')
        mean.fit(X=points)
        result = mean.estimate_
        expected = point
        self.assertAllClose(expected, result)

    @geomstats.tests.np_and_tf_only
    def test_variance_hyperbolic(self):
        point = gs.array([2., 1., 1., 1.])
        points = gs.array([point, point])
        result = variance(
            points, base_point=point, metric=self.hyperbolic.metric)
        expected = gs.array(0.)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_estimate_hyperbolic(self):
        point = gs.array([2., 1., 1., 1.])
        points = gs.array([point, point])

        mean = FrechetMean(metric=self.hyperbolic.metric)
        mean.fit(X=points)
        expected = point

        result = mean.estimate_

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_estimate_and_belongs_hyperbolic(self):
        point_a = self.hyperbolic.random_uniform()
        point_b = self.hyperbolic.random_uniform()
        point_c = self.hyperbolic.random_uniform()
        points = gs.stack([point_a, point_b, point_c], axis=0)

        mean = FrechetMean(metric=self.hyperbolic.metric)
        mean.fit(X=points)

        result = self.hyperbolic.belongs(mean.estimate_)
        expected = True

        self.assertAllClose(result, expected)

    def test_mean_euclidean_shape(self):
        dim = 2
        point = gs.array([1., 4.])

        mean = FrechetMean(metric=self.euclidean.metric)
        points = [point, point, point]
        mean.fit(points)

        result = mean.estimate_

        self.assertAllClose(gs.shape(result), (dim,))

    def test_mean_euclidean(self):
        point = gs.array([1., 4.])

        mean = FrechetMean(metric=self.euclidean.metric)
        points = [point, point, point]
        mean.fit(points)

        result = mean.estimate_
        expected = point

        self.assertAllClose(result, expected)

        points = gs.array([
            [1., 2.],
            [2., 3.],
            [3., 4.],
            [4., 5.]])
        weights = gs.array([1., 2., 1., 2.])

        mean = FrechetMean(metric=self.euclidean.metric)
        mean.fit(points, weights=weights)

        result = mean.estimate_
        expected = gs.array([16. / 6., 22. / 6.])

        self.assertAllClose(result, expected)

    def test_variance_euclidean(self):
        points = gs.array([
            [1., 2.],
            [2., 3.],
            [3., 4.],
            [4., 5.]])
        weights = gs.array([1., 2., 1., 2.])
        base_point = gs.zeros(2)
        result = variance(
            points, weights=weights, base_point=base_point,
            metric=self.euclidean.metric)
        # we expect the average of the points' sq norms.
        expected = gs.array((1 * 5. + 2 * 13. + 1 * 25. + 2 * 41.) / 6.)

        self.assertAllClose(result, expected)

    def test_mean_matrices_shape(self):
        m, n = (2, 2)
        point = gs.array([
            [1., 4.],
            [2., 3.]])

        metric = MatricesMetric(m, n)
        mean = FrechetMean(metric=metric, point_type='matrix')
        points = [point, point, point]
        mean.fit(points)

        result = mean.estimate_

        self.assertAllClose(gs.shape(result), (m, n))

    def test_mean_matrices(self):
        m, n = (2, 2)
        point = gs.array([
            [1., 4.],
            [2., 3.]])

        metric = MatricesMetric(m, n)
        mean = FrechetMean(metric=metric, point_type='matrix')
        points = [point, point, point]
        mean.fit(points)

        result = mean.estimate_
        expected = point

        self.assertAllClose(result, expected)

    def test_mean_minkowski_shape(self):
        dim = 2
        point = gs.array([2., -math.sqrt(3)])
        points = [point, point, point]

        mean = FrechetMean(metric=self.minkowski.metric)
        mean.fit(points)
        result = mean.estimate_

        self.assertAllClose(gs.shape(result), (dim,))

    def test_mean_minkowski(self):
        point = gs.array([2., -math.sqrt(3)])
        points = [point, point, point]

        mean = FrechetMean(metric=self.minkowski.metric)
        mean.fit(points)
        result = mean.estimate_

        expected = point

        self.assertAllClose(result, expected)

        points = gs.array([
            [1., 0.],
            [2., math.sqrt(3)],
            [3., math.sqrt(8)],
            [4., math.sqrt(24)]])
        weights = gs.array([1., 2., 1., 2.])

        mean = FrechetMean(metric=self.minkowski.metric)
        mean.fit(points, weights=weights)
        result = mean.estimate_
        result = self.minkowski.belongs(result)
        expected = gs.array(True)

        self.assertAllClose(result, expected)

    def test_variance_minkowski(self):
        points = gs.array([
            [1., 0.],
            [2., math.sqrt(3)],
            [3., math.sqrt(8)],
            [4., math.sqrt(24)]])
        weights = gs.array([1., 2., 1., 2.])
        base_point = gs.array([-1., 0.])
        var = variance(
            points, weights=weights, base_point=base_point,
            metric=self.minkowski.metric)
        result = var != 0
        # we expect the average of the points' Minkowski sq norms.
        expected = True
        self.assertAllClose(result, expected)
