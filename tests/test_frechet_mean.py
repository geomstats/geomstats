"""Unit tests for Frechet mean."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean


class TestFrechetMean(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.sphere = Hypersphere(dim=2)
        self.hyperbolic = Hyperbolic(dim=2)

    @geomstats.tests.np_only
    def test_adaptive_gradient_descent_sphere(self):
        n_tests = 100
        result = gs.zeros(n_tests)
        expected = gs.zeros(n_tests)

        for i in range(n_tests):
            # take 2 random points, compute their mean, and verify that
            # log of each at the mean is opposite
            points = self.sphere.random_uniform(n_samples=2)
            mean = _adaptive_gradient_descent(points=points, metric=self.sphere.metric)

            logs = self.sphere.metric.log(point=points, base_point=mean)
            result[i] = gs.linalg.norm(logs[1, :] + logs[0, :])

        self.assertAllClose(expected, result, rtol=1e-10, atol=1e-10)

    @geomstats.tests.np_and_pytorch_only
    def test_mean_and_belongs_sphere(self):
        point_a = gs.array([1., 0., 0., 0., 0.])
        point_b = gs.array([0., 1., 0., 0., 0.])
        points = gs.zeros((2, point_a.shape[0]))
        points[0, :] = point_a
        points[1, :] = point_b

        mean = FrechetMean(points, metric=self.sphere.metric)

        result = self.space.belongs(mean.mean_)
        expected = gs.array([[True]])
        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_pytorch_only
    def test_variance_sphere(self):
        point = gs.array([0., 0., 0., 0., 1.])
        points = gs.zeros((2, point.shape[0]))
        points[0, :] = point
        points[1, :] = point
        result = self.metric.variance(points)
        expected = helper.to_scalar(0.)

        self.assertAllClose(expected, result)

    @geomstats.tests.np_and_pytorch_only
    def test_mean_sphere(self):
        point = gs.array([0., 0., 0., 0., 1.])
        points = gs.zeros((2, point.shape[0]))
        points[0, :] = point
        points[1, :] = point

        mean = FrechetMean(metric=self.metric)

        mean.fit(X=points)
        result = mean.mean_
        expected = point

        self.assertAllClose(expected, result)

    @geomstats.tests.np_and_tf_only
    def test_variance_hyperbolic(self):
        point = gs.array([2., 1., 1., 1.])
        points = gs.array([point, point])
        result = self.metric.variance(points)
        expected = helper.to_scalar(0.)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_mean_hyperbolic(self):
        point = gs.array([2., 1., 1., 1.])
        points = gs.array([point, point])
        result = self.metric.mean(points)
        expected = helper.to_vector(point)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_mean_and_belongs_hyperbolic(self):
        point_a = self.space.random_uniform()
        point_b = self.space.random_uniform()
        point_c = self.space.random_uniform()
        points = gs.concatenate([point_a, point_b, point_c], axis=0)

        mean = self.metric.mean(points)
        result = self.space.belongs(mean)
        expected = gs.array([[True]])

        self.assertAllClose(result, expected)


