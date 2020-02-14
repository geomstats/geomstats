"""Unit tests for Frechet mean."""

import math

import tests.helper as helper

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.minkowski import Minkowski
from geomstats.learning.frechet_mean import _adaptive_gradient_descent
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.frechet_mean import variance


class TestFrechetMean(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.sphere = Hypersphere(dimension=4)
        self.hyperbolic = Hyperbolic(dimension=3)
        self.euclidean = Euclidean(dimension=2)
        self.minkowski = Minkowski(dimension=2)

    @geomstats.tests.np_only
    def test_adaptive_gradient_descent_sphere(self):
        n_tests = 100
        result = gs.zeros(n_tests)
        expected = gs.zeros(n_tests)

        for i in range(n_tests):
            # take 2 random points, compute their mean, and verify that
            # log of each at the mean is opposite
            points = self.sphere.random_uniform(n_samples=2)
            mean = _adaptive_gradient_descent(
                points=points, metric=self.sphere.metric)

            logs = self.sphere.metric.log(point=points, base_point=mean)
            result[i] = gs.linalg.norm(logs[1, :] + logs[0, :])

        self.assertAllClose(expected, result, rtol=1e-10, atol=1e-10)

    @geomstats.tests.np_and_pytorch_only
    def test_estimate_and_belongs_sphere(self):
        point_a = gs.array([1., 0., 0., 0., 0.])
        point_b = gs.array([0., 1., 0., 0., 0.])
        points = gs.zeros((2, point_a.shape[0]))
        points[0, :] = point_a
        points[1, :] = point_b

        mean = FrechetMean(metric=self.sphere.metric)
        mean.fit(points)

        result = self.sphere.belongs(mean.estimate_)
        expected = gs.array([[True]])
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_variance_sphere(self):
        point = gs.array([0., 0., 0., 0., 1.])
        points = gs.zeros((2, point.shape[0]))
        points[0, :] = point
        points[1, :] = point

        result = variance(
            points, base_point=point, metric=self.sphere.metric)
        expected = helper.to_scalar(0.)

        self.assertAllClose(expected, result)

    @geomstats.tests.np_and_pytorch_only
    def test_estimate_sphere(self):
        point = gs.array([0., 0., 0., 0., 1.])
        points = gs.zeros((2, point.shape[0]))
        points[0, :] = point
        points[1, :] = point

        mean = FrechetMean(metric=self.sphere.metric)
        mean.fit(X=points)

        result = mean.estimate_
        expected = helper.to_vector(point)

        self.assertAllClose(expected, result)

    @geomstats.tests.np_and_tf_only
    def test_variance_hyperbolic(self):
        point = gs.array([2., 1., 1., 1.])
        points = gs.array([point, point])
        result = variance(
            points, base_point=point, metric=self.hyperbolic.metric)
        expected = helper.to_scalar(0.)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_estimate_hyperbolic(self):
        point = gs.array([2., 1., 1., 1.])
        points = gs.array([point, point])

        mean = FrechetMean(metric=self.hyperbolic.metric)
        mean.fit(X=points)

        result = mean.estimate_
        expected = helper.to_vector(point)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_estimate_and_belongs_hyperbolic(self):
        point_a = self.hyperbolic.random_uniform()
        point_b = self.hyperbolic.random_uniform()
        point_c = self.hyperbolic.random_uniform()
        points = gs.concatenate([point_a, point_b, point_c], axis=0)

        mean = FrechetMean(metric=self.hyperbolic.metric)
        mean.fit(X=points)

        result = self.hyperbolic.belongs(mean.estimate_)
        expected = gs.array([[True]])

        self.assertAllClose(result, expected)

    def test_mean_euclidean(self):
        point = gs.array([[1., 4.]])

        mean = FrechetMean(metric=self.euclidean.metric)
        points = [point, point, point]
        mean.fit(points)

        result = mean.estimate_
        expected = point
        expected = helper.to_vector(expected)

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
        expected = helper.to_vector(expected)

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
        expected = (1 * 5. + 2 * 13. + 1 * 25. + 2 * 41.) / 6.
        expected = helper.to_scalar(expected)

        self.assertAllClose(result, expected)

    def test_mean_minkowski(self):
        point = gs.array([[2., -math.sqrt(3)]])
        points = [point, point, point]

        mean = FrechetMean(metric=self.minkowski.metric)
        mean.fit(points)
        result = mean.estimate_

        expected = point
        expected = helper.to_vector(expected)

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
        expected = gs.array([[True]])

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
        result = helper.to_scalar(var != 0)
        # we expect the average of the points' Minkowski sq norms.
        expected = helper.to_scalar(gs.array([True]))
        self.assertAllClose(result, expected)
