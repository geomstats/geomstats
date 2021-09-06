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
        self.n_samples = 20

        # Set up for hypersphere
        self.dim_sphere = 4
        self.shape_sphere = (self.dim_sphere + 1,)
        self.sphere = Hypersphere(dim=self.dim_sphere)
        times = gs.random.rand(self.n_samples)
        self.times_sphere = times - gs.mean(times)
        self.intercept_sphere_true = self.sphere.random_point()
        self.coef_sphere_true = self.sphere.projection(
            gs.random.rand(self.dim_sphere+1))

        self.target_sphere = self.sphere.metric.exp(
            self.times_sphere[:, None] * self.coef_sphere_true, 
            base_point=self.intercept_sphere_true)
        
        self.parameter_sphere_true = gs.vstack(
            [self.intercept_sphere_true, 
            self.coef_sphere_true])
        self.parameter_sphere_guess = gs.vstack(
            [self.target_sphere[0], 
            self.sphere.to_tangent(
                gs.random.normal(size=self.shape_sphere), 
                self.target_sphere[0])])
        print("\n\n\nWOOOO")
        print(self.parameter_sphere_true.shape)
        print(self.parameter_sphere_guess.shape)

        print("\n\n\n dtypes")
        print(
            self.times_sphere.dtype,
            self.target_sphere.dtype, 
            self.parameter_sphere_guess.dtype,
            self.parameter_sphere_true)
        print("\n\n\n\n")

        # Set up for special euclidean
        self.se2 = SpecialEuclidean(n=2)
        self.metric_se2 = self.se2.left_canonical_metric
        self.metric_se2.default_point_type = 'matrix'

        self.shape_se2 = (3, 3)
        #times = gs.random.rand(self.n_samples)
        times = gs.linspace(0., 1., self.n_samples) 
        self.times_se2 = times - gs.mean(times)

        self.intercept_se2_true = self.se2.identity #self.se2.random_point()
        vector = gs.array([
            [1., 4., 3.],
            [-1., 2., -3],
            [0., -1., 1.]
        ])
        self.coef_se2_true = self.se2.to_tangent(
            vector, 
            self.intercept_se2_true)
        # self.coef_se2_true = self.se2.to_tangent(
        #     5. * gs.random.rand(*self.shape_se2), 
        #     self.intercept_se2_true)

        self.target_se2 = self.metric_se2.exp(
            self.times_se2[:, None, None] * self.coef_se2_true[None], 
            self.intercept_se2_true)

        self.parameter_se2_true = gs.vstack([
            gs.flatten(self.intercept_se2_true), 
            gs.flatten(self.coef_se2_true)])
        self.parameter_se2_guess = gs.vstack([
            gs.flatten(self.target_se2[0]), 
            gs.flatten(self.se2.to_tangent(
                gs.random.normal(size=self.shape_se2), 
                self.target_se2[0]))])

    def test_loss_hypersphere(self):
        gr = GeodesicRegression(
            self.sphere, metric=self.sphere.metric, center_data=False, algorithm='extrinsic',
            verbose=True, max_iter=50, learning_rate=0.1)
        loss = gr._loss(
            self.times_sphere, 
            self.target_sphere, 
            self.parameter_sphere_true, 
            self.shape_sphere)
        self.assertAllClose(loss.shape, ())
        self.assertTrue(gs.isclose(loss, 0.))

        gr = GeodesicRegression(
            self.sphere, metric=self.sphere.metric, center_data=False, algorithm='riemannian',
            verbose=True, max_iter=50, learning_rate=0.1)
        loss = gr._loss(
            self.times_sphere, 
            self.target_sphere, 
            self.parameter_sphere_true, 
            self.shape_sphere)
        self.assertAllClose(loss.shape, ())
        self.assertTrue(gs.isclose(loss, 0.))


    def test_value_and_grad_loss_hypersphere(self):
        gr = GeodesicRegression(
            self.sphere, metric=self.sphere.metric, center_data=False, algorithm='extrinsic',
            verbose=True, max_iter=50, learning_rate=0.1)

        def loss_of_param(param):
            return gr._loss(
                self.times_sphere, 
                self.target_sphere, 
                param, 
                self.shape_sphere)

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param)
        loss_value, loss_grad = objective_with_grad(
            self.parameter_sphere_guess)
        print("\n\nshape of input")
        print(self.parameter_sphere_guess.shape)

        expected_grad_shape = (2, self.dim_sphere + 1)
        self.assertAllClose(loss_value.shape, ())
        self.assertAllClose(loss_grad.shape, expected_grad_shape)

        self.assertFalse(gs.isclose(loss_value, 0.))
        self.assertFalse(gs.isnan(loss_value))
        self.assertFalse(
            gs.all(gs.isclose(loss_grad, gs.zeros(expected_grad_shape))))
        self.assertTrue(gs.all(~gs.isnan(loss_grad)))

        objective_with_grad = gs.autodiff.value_and_grad(
            loss_of_param, to_numpy=True)
        loss_value, loss_grad = objective_with_grad(
            self.parameter_sphere_guess)

        expected_grad_shape = (2, self.dim_sphere + 1)
        self.assertAllClose(loss_value.shape, ())
        self.assertAllClose(loss_grad.shape, expected_grad_shape)

        self.assertFalse(gs.isclose(loss_value, 0.))
        self.assertFalse(gs.isnan(loss_value))
        self.assertFalse(
            gs.all(gs.isclose(loss_grad, gs.zeros(expected_grad_shape))))
        self.assertTrue(gs.all(~gs.isnan(loss_grad)))

    def test_loss_minimization_extrinsic_hypersphere(self):
        """Minimize loss from noiseless data."""
        gr = GeodesicRegression(
            self.sphere, metric=self.sphere.metric, center_data=False, algorithm='extrinsic',
            verbose=True, max_iter=50, learning_rate=0.1)

        def loss_of_param(param):
            return gr._loss(
                self.times_sphere, 
                self.target_sphere, 
                param, 
                self.shape_sphere)

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param, to_numpy=True)
        res = minimize(
            objective_with_grad, gs.flatten(self.parameter_sphere_guess), method='CG', jac=True,
            options={'disp': True, 'maxiter': 50})
        self.assertAllClose(
            gs.array(res.x).shape, ((self.dim_sphere + 1) * 2,))
        self.assertTrue(gs.isclose(res.fun, 0., atol=100 * gs.atol))

        # Cast required because minimization happens in scipy in float64
        param_hat = gs.cast(
                gs.array(res.x), 
                self.parameter_sphere_true.dtype)

        intercept_hat, coef_hat = gs.split(param_hat, 2)
        intercept_hat = self.sphere.projection(intercept_hat)
        coef_hat = self.sphere.to_tangent(coef_hat, intercept_hat)
        self.assertAllClose(intercept_hat, self.intercept_sphere_true, atol=1e-3)

        tangent_vec_of_transport = self.sphere.metric.log(
            self.intercept_sphere_true, base_point=intercept_hat)

        transported_coef_hat = self.sphere.metric.parallel_transport(
            tangent_vec_a=coef_hat,
            tangent_vec_b=tangent_vec_of_transport,
            base_point=intercept_hat)

        self.assertAllClose(transported_coef_hat, self.coef_sphere_true, atol=2*1e-1)

    def test_fit_extrinsic_hypersphere(self):
        gr = GeodesicRegression(
            self.sphere, metric=self.sphere.metric, center_data=False, algorithm='extrinsic',
            verbose=True, max_iter=50, learning_rate=0.1)

        gr.fit(self.times_sphere, self.target_sphere, compute_training_score=True)

        training_score = gr.training_score_
        intercept_hat, coef_hat = gr.intercept_, gr.coef_
        self.assertAllClose(intercept_hat.shape, self.shape_sphere)
        self.assertAllClose(coef_hat.shape, self.shape_sphere)
        self.assertAllClose(training_score, 1., atol=500 * gs.atol)
        self.assertAllClose(intercept_hat, self.intercept_sphere_true, atol=5e3 * gs.atol)

        tangent_vec_of_transport = self.sphere.metric.log(
            self.intercept_sphere_true, base_point=intercept_hat)

        transported_coef_hat = self.sphere.metric.parallel_transport(
            tangent_vec_a=coef_hat,
            tangent_vec_b=tangent_vec_of_transport,
            base_point=intercept_hat)

        self.assertAllClose(transported_coef_hat, self.coef_sphere_true, atol=2*1e-1)
    
    def test_fit_riemannian_hypersphere(self):
        gr = GeodesicRegression(
            self.sphere, metric=self.sphere.metric, center_data=False, algorithm='riemannian',
            verbose=True, max_iter=50, learning_rate=0.1)

        gr.fit(self.times_sphere, self.target_sphere, compute_training_score=True)
        intercept_hat, coef_hat = gr.intercept_, gr.coef_
        training_score = gr.training_score_

        self.assertAllClose(intercept_hat.shape, self.shape_sphere)
        self.assertAllClose(coef_hat.shape, self.shape_sphere)

        self.assertAllClose(training_score, 1., atol=5e3 * gs.atol)
        self.assertAllClose(intercept_hat, self.intercept_sphere_true, atol=5e3 * gs.atol)

        tangent_vec_of_transport = self.sphere.metric.log(
            self.intercept_sphere_true, base_point=intercept_hat)

        transported_coef_hat = self.sphere.metric.parallel_transport(
            tangent_vec_a=coef_hat,
            tangent_vec_b=tangent_vec_of_transport,
            base_point=intercept_hat)

        self.assertAllClose(transported_coef_hat, self.coef_sphere_true, atol=2*1e-1)

    def test_loss_minimization_extrinsic_se2(self):
        gr = GeodesicRegression(
            self.se2, metric=self.metric_se2, center_data=False, algorithm='extrinsic',
            verbose=True, max_iter=50, learning_rate=0.1)
 
        def loss_of_param(param):
            return gr._loss(
                self.times_se2, 
                self.target_se2, 
                param, 
                self.shape_se2)

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param, to_numpy=True)
        
        res = minimize(
            objective_with_grad, gs.flatten(self.parameter_se2_guess), method='CG', jac=True,
            options={'disp': True, 'maxiter': 50})
        self.assertAllClose(gs.array(res.x).shape, (18,))

        self.assertTrue(gs.isclose(res.fun, 0.))

        intercept_hat, coef_hat = gs.split(gs.array(res.x), 2)
        intercept_hat = gs.cast(intercept_hat, dtype=self.target_se2.dtype)
        coef_hat = gs.cast(coef_hat, dtype=self.target_se2.dtype)
        intercept_hat = gs.reshape(intercept_hat, self.shape_se2)
        coef_hat = gs.reshape(coef_hat, self.shape_se2)
        intercept_hat = self.se2.projection(intercept_hat)
        coef_hat = self.se2.to_tangent(coef_hat, intercept_hat)
        self.assertAllClose(intercept_hat, self.intercept_se2_true, atol=1e-4)
        self.assertAllClose(coef_hat, self.coef_se2_true, atol=1e-4)

    def test_fit_extrinsic_se2(self):
        gr = GeodesicRegression(
            self.se2, metric=self.metric_se2, center_data=False, algorithm='extrinsic',
            verbose=True, max_iter=50, learning_rate=0.1)

        gr.fit(self.times_se2, self.target_se2, compute_training_score=True)
        intercept_hat, coef_hat = gr.intercept_, gr.coef_
        training_score = gr.training_score_

        self.assertAllClose(intercept_hat.shape, self.shape_se2)
        self.assertAllClose(coef_hat.shape, self.shape_se2)
        # print("TRAINING SCORE SE@ EXTR")
        # print(training_score)
        self.assertTrue(gs.isclose(training_score, 1.))
        self.assertAllClose(intercept_hat, self.intercept_se2_true)
        self.assertAllClose(coef_hat, self.coef_se2_true)

    def test_fit_riemannian_se2(self):
        gr = GeodesicRegression(
            self.se2, metric=self.metric_se2, center_data=False, algorithm='riemannian',
            verbose=True, max_iter=50, learning_rate=0.1)

        gr.fit(self.times_se2, self.target_se2)
        intercept_hat, coef_hat = gr.intercept_, gr.coef_
        self.assertAllClose(intercept_hat.shape, self.shape_se2)
        self.assertAllClose(coef_hat.shape, self.shape_se2)
        
    def test_value_and_grad_loss_se2(self):

        gr = GeodesicRegression(
            self.se2, metric=self.metric_se2, center_data=False, algorithm='extrinsic',
            verbose=True, max_iter=50, learning_rate=0.1)

        def loss_of_param(param):
            return gr._loss(
                self.times_se2, 
                self.target_se2, 
                param, 
                self.shape_se2)

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param)
        loss_value, loss_grad = objective_with_grad(self.parameter_se2_true)
        expected_grad_shape = (2 * self.shape_se2[0] * self.shape_se2[1],)

        self.assertTrue(gs.isclose(loss_value, 0.))
        self.assertTrue(
            gs.all(gs.isclose(loss_grad, gs.zeros(expected_grad_shape), atol=1e-5)))

        loss_value, loss_grad = objective_with_grad(self.parameter_se2_guess)
        
        self.assertAllClose(loss_value.shape, ())
        self.assertAllClose(loss_grad.shape, expected_grad_shape)

        self.assertFalse(gs.isclose(loss_value, 0.))
        self.assertFalse(gs.isnan(loss_value))
        self.assertFalse(
            gs.all(gs.isclose(loss_grad, gs.zeros(expected_grad_shape))))
        self.assertTrue(gs.all(~gs.isnan(loss_grad)))

        objective_with_grad = gs.autodiff.value_and_grad(
            loss_of_param, to_numpy=True)
        loss_value, loss_grad = objective_with_grad(self.parameter_se2_guess)
        expected_grad_shape = (2 * self.shape_se2[0] * self.shape_se2[1],)
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
            self.times_se2, 
            self.target_se2, 
            self.parameter_se2_true, 
            self.shape_se2)
        self.assertAllClose(loss.shape, ())
        self.assertTrue(gs.isclose(loss, 0.))

        gr = GeodesicRegression(
            self.se2, metric=self.metric_se2, center_data=False, algorithm='riemannian',
            verbose=True, max_iter=50, learning_rate=0.1)
        loss = gr._loss(
            self.times_se2, 
            self.target_se2, 
            self.parameter_se2_true, 
            self.shape_se2)
        self.assertAllClose(loss.shape, ())
        self.assertTrue(gs.isclose(loss, 0.))
