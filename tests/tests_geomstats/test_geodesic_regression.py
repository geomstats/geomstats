"""Unit tests for Geodesic Regression."""

from scipy.optimize import minimize

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
        self.shape_sphere = (self.dim_sphere + 1,)
        self.sphere = Hypersphere(dim=self.dim_sphere)
        input_data = gs.random.rand(self.n_samples)
        self.input_data_sphere = input_data - gs.mean(input_data)
        target = gs.random.rand(self.n_samples, self.dim_sphere + 1)
        self.target_sphere = self.sphere.projection(target)
        intercept = self.sphere.random_point()
        coef = self.sphere.projection(
            gs.random.rand(self.dim_sphere+1))
        self.parameter_sphere = gs.vstack([intercept, coef])

        self.se2 = SpecialEuclidean(n=2)
        self.metric_se2 = self.se2.left_canonical_metric
        self.metric_se2.default_point_type = 'matrix'

        self.shape_se2 = (3, 3)
        input_data = gs.random.rand(self.n_samples)
        self.input_data_se2 = input_data - gs.mean(input_data)

        intercept = self.se2.random_point()
        coef = self.se2.to_tangent(5. * gs.random.rand(*self.shape_se2), intercept)

        target = self.metric_se2.exp(
            self.input_data_se2[:, None, None] * coef[None], intercept)
        normal_noise = gs.random.normal(size=(self.n_samples,) + self.shape_se2)
        noise = self.se2.to_tangent(normal_noise, target) / gs.pi / 2
        self.target_se2 = self.metric_se2.exp(noise, target)
        self.parameter_se2 = gs.vstack([intercept, coef])

    def test_loss_hypersphere(self):
        gr = GeodesicRegression(
            self.sphere, metric=self.sphere.metric, center_data=False, algorithm='extrinsic',
            verbose=True, max_iter=50, learning_rate=0.1)
        loss = gr._loss(
            self.input_data_sphere, 
            self.target_sphere, 
            self.parameter_sphere, 
            self.shape_sphere)
        self.assertAllClose(loss.shape, ())

        gr = GeodesicRegression(
            self.sphere, metric=self.sphere.metric, center_data=False, algorithm='riemannian',
            verbose=True, max_iter=50, learning_rate=0.1)
        loss = gr._loss(
            self.input_data_sphere, 
            self.target_sphere, 
            self.parameter_sphere, 
            self.shape_sphere)
        self.assertAllClose(loss.shape, ())


    def test_value_and_grad_loss_hypersphere(self):
        gr = GeodesicRegression(
            self.sphere, metric=self.sphere.metric, center_data=False, algorithm='extrinsic',
            verbose=True, max_iter=50, learning_rate=0.1)

        def loss_of_param(param):
            return gr._loss(
                self.input_data_sphere, 
                self.target_sphere, 
                param, 
                self.shape_sphere)

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param)
        loss_value, loss_grad = objective_with_grad(self.parameter_sphere)

        expected_grad_shape = (self.dim_sphere + 1) * 2,
        self.assertAllClose(loss_value.shape, ())
        self.assertAllClose(loss_grad.shape, expected_grad_shape)

        self.assertFalse(gs.isclose(loss_value, 0.))
        self.assertFalse(gs.isnan(loss_value))
        self.assertFalse(
            gs.all(gs.isclose(loss_grad, gs.zeros(expected_grad_shape))))
        self.assertTrue(gs.all(~gs.isnan(loss_grad)))

        objective_with_grad = gs.autodiff.value_and_grad(
            loss_of_param, to_numpy=True)
        loss_value, loss_grad = objective_with_grad(self.parameter_sphere)

        expected_grad_shape = (self.dim_sphere + 1) * 2,
        self.assertAllClose(loss_value.shape, ())
        self.assertAllClose(loss_grad.shape, expected_grad_shape)

        self.assertFalse(gs.isclose(loss_value, 0.))
        self.assertFalse(gs.isnan(loss_value))
        self.assertFalse(
            gs.all(gs.isclose(loss_grad, gs.zeros(expected_grad_shape))))
        self.assertTrue(gs.all(~gs.isnan(loss_grad)))

    def test_loss_minimization_extrinsic_hypersphere(self):
        gr = GeodesicRegression(
            self.sphere, metric=self.sphere.metric, center_data=False, algorithm='extrinsic',
            verbose=True, max_iter=50, learning_rate=0.1)

        def loss_of_param(param):
            return gr._loss(
                self.input_data_sphere, 
                self.target_sphere, 
                param, 
                self.shape_sphere)

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param, to_numpy=True)
        res = minimize(
            objective_with_grad, gs.flatten(self.parameter_sphere), method='CG', jac=True,
            options={'disp': True, 'maxiter': 10})
        self.assertAllClose(gs.array(res.x).shape, ((self.dim_sphere + 1) * 2,))
        
    def test_loss_minimization_riemannian_hypersphere(self):
        gr = GeodesicRegression(
            self.sphere, metric=self.sphere.metric, center_data=False, algorithm='riemannian',
            verbose=True, max_iter=50, learning_rate=0.1)

        def loss_of_param(param):
            return gr._loss(
                self.input_data_sphere, 
                self.target_sphere, 
                param, 
                self.shape_sphere)

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param, to_numpy=True)
        res = minimize(
            objective_with_grad, gs.flatten(self.parameter_sphere), method='CG', jac=True,
            options={'disp': True, 'maxiter': 10})
        self.assertAllClose(gs.array(res.x).shape, ((self.dim_sphere + 1) * 2,))

    def test_loss_minimization_extrinsic_se2(self):
        gr = GeodesicRegression(
            self.se2, metric=self.metric_se2, center_data=False, algorithm='extrinsic',
            verbose=True, max_iter=50, learning_rate=0.1)
 
        def loss_of_param(param):
            return gr._loss(
                self.input_data_se2, 
                self.target_se2, 
                param, 
                self.shape_se2)

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param, to_numpy=True)
        print(self.parameter_se2.shape)
        res = minimize(
            objective_with_grad, gs.flatten(self.parameter_se2), method='CG', jac=True,
            options={'disp': True, 'maxiter': 10})
        self.assertAllClose(gs.array(res.x).shape, (18,))

    def test_loss_minimization_riemannian_se2(self):
        gr = GeodesicRegression(
            self.se2, metric=self.metric_se2, center_data=False, algorithm='riemannian',
            verbose=True, max_iter=50, learning_rate=0.1)
 
        def loss_of_param(param):
            return gr._loss(
                self.input_data_se2, 
                self.target_se2, 
                param, 
                self.shape_se2)

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param, to_numpy=True)
        print(self.parameter_se2.shape)
        res = minimize(
            objective_with_grad, gs.flatten(self.parameter_se2), method='CG', jac=True,
            options={'disp': True, 'maxiter': 10})
        self.assertAllClose(gs.array(res.x).shape, (18,))
        
    def test_value_and_grad_loss_se2(self):

        gr = GeodesicRegression(
            self.se2, metric=self.metric_se2, center_data=False, algorithm='extrinsic',
            verbose=True, max_iter=50, learning_rate=0.1)

        def loss_of_param(param):
            return gr._loss(
                self.input_data_se2, 
                self.target_se2, 
                param, 
                self.shape_se2)

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param)
        loss_value, loss_grad = objective_with_grad(self.parameter_se2)
        expected_grad_shape = (2 * self.shape_se2[0], self.shape_se2[1])
        self.assertAllClose(loss_value.shape, ())
        self.assertAllClose(loss_grad.shape, expected_grad_shape)

        self.assertFalse(gs.isclose(loss_value, 0.))
        self.assertFalse(gs.isnan(loss_value))
        self.assertFalse(
            gs.all(gs.isclose(loss_grad, gs.zeros(expected_grad_shape))))
        self.assertTrue(gs.all(~gs.isnan(loss_grad)))

        objective_with_grad = gs.autodiff.value_and_grad(
            loss_of_param, to_numpy=True)
        loss_value, loss_grad = objective_with_grad(self.parameter_se2)
        expected_grad_shape = (2 * self.shape_se2[0], self.shape_se2[1])
        self.assertAllClose(loss_value.shape, ())
        self.assertAllClose(loss_grad.shape, expected_grad_shape)

        self.assertFalse(gs.isclose(loss_value, 0.))
        self.assertFalse(gs.isnan(loss_value))
        self.assertFalse(
            gs.all(gs.isclose(loss_grad, gs.zeros(expected_grad_shape))))
        self.assertTrue(gs.all(~gs.isnan(loss_grad)))

    def test_loss_se2(self):
        gr = GeodesicRegression(
            self.se2, metric=self.metric_se2, center_data=False, algorithm='extrinsic',
            verbose=True, max_iter=50, learning_rate=0.1)
        loss = gr._loss(
            self.input_data_se2, 
            self.target_se2, 
            self.parameter_se2, 
            self.shape_se2)
        self.assertAllClose(loss.shape, ())

        gr = GeodesicRegression(
            self.se2, metric=self.metric_se2, center_data=False, algorithm='riemannian',
            verbose=True, max_iter=50, learning_rate=0.1)
        loss = gr._loss(
            self.input_data_se2, 
            self.target_se2, 
            self.parameter_se2, 
            self.shape_se2)
        self.assertAllClose(loss.shape, ())
