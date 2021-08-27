"""Unit tests for Geodesic Regression."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.learning.geodesic_regression import GeodesicRegression



class TestGeodesicRegression(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(123)
        self.n_samples = 3
        self.dim_sphere = 4
        self.sphere = Hypersphere(dim=self.dim_sphere)
        self.se2 = SpecialEuclidean(n=2)

    def test_loss_hypersphere(self):
        input_data = gs.random.rand(self.n_samples)
        input_data -= gs.mean(input_data)

        target = gs.random.rand(self.n_samples, self.dim_sphere + 1)
        target = self.sphere.projection(target)

        intercept = self.sphere.random_point()
        coef = self.sphere.projection(
            gs.random.rand(self.dim_sphere+1))
        parameter = (intercept, coef)

        shape = (self.dim_sphere + 1,)

        gr = GeodesicRegression(
            self.sphere, metric=self.sphere.metric, center_data=False, algorithm='extrinsic',
            verbose=True, max_iter=50, learning_rate=0.1)
        loss = gr._loss(input_data, target, parameter, shape)
        self.assertAllClose(loss.shape, ())

        gr = GeodesicRegression(
            self.sphere, metric=self.sphere.metric, center_data=False, algorithm='riemannian',
            verbose=True, max_iter=50, learning_rate=0.1)
        loss = gr._loss(input_data, target, parameter, shape)
        self.assertAllClose(loss.shape, ())

    def test_loss_se2(self):
        metric = self.se2.left_canonical_metric
        metric.default_point_type = 'matrix'
        shape = (3, 3)
        gs.random.seed(0)

        # Generate data
        input_data = gs.random.rand(self.n_samples)
        input_data -= gs.mean(input_data)

        intercept = self.se2.random_point()
        coef = self.se2.to_tangent(5. * gs.random.rand(*shape), intercept)

        target = metric.exp(input_data[:, None, None] * coef[None], intercept)

        parameter = (intercept, coef)

        gr = GeodesicRegression(
            self.se2, metric=metric, center_data=False, algorithm='extrinsic',
            verbose=True, max_iter=50, learning_rate=0.1)
        loss = gr._loss(input_data, target, parameter, shape)
        self.assertAllClose(loss.shape, ())


        gr = GeodesicRegression(
            self.se2, metric=metric, center_data=False, algorithm='riemannian',
            verbose=True, max_iter=50, learning_rate=0.1)
        loss = gr._loss(input_data, target, parameter, shape)
        self.assertAllClose(loss.shape, ())
