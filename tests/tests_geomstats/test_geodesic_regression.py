"""Unit tests for Geodesic Regression."""

import math

from scipy.optimize import minimize

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.discrete_curves import DiscreteCurves
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.learning.geodesic_regression import GeodesicRegression


class TestGeodesicRegression(tests.conftest.TestCase):
    _multiprocess_can_split_ = True

    def setup_method(self):
        gs.random.seed(1234)
        self.n_samples = 20

        # Set up for euclidean
        self.eucl = Euclidean(dim=3)

        X = gs.random.rand(self.n_samples)

        self.intercept_eucl_true = self.eucl.random_point()
        self.coef_eucl_true = self.eucl.random_point()
        self.param_eucl_true = gs.vstack(
            [self.intercept_eucl_true, self.coef_eucl_true]
        )

        self.X_eucl = X - gs.mean(X)
        self.y_eucl = (
            self.intercept_eucl_true + self.X_eucl[:, None] * self.coef_eucl_true
        )

        self.param_eucl_guess = gs.vstack(
            [self.y_eucl[0], self.y_eucl[0] + gs.random.normal(size=self.eucl.shape)]
        )

        # Set up for hypersphere
        self.sphere = Hypersphere(dim=4)

        self.intercept_sphere_true = self.sphere.random_point()
        # TODO: can go with a random point
        self.coef_sphere_true = self.sphere.projection(
            gs.random.rand(self.sphere.dim + 1)
        )
        self.param_sphere_true = gs.vstack(
            [self.intercept_sphere_true, self.coef_sphere_true]
        )

        X = gs.random.rand(self.n_samples)
        self.X_sphere = X - gs.mean(X)
        self.y_sphere = self.sphere.metric.exp(
            self.X_sphere[:, None] * self.coef_sphere_true,
            base_point=self.intercept_sphere_true,
        )

        self.param_sphere_guess = gs.vstack(
            [
                self.y_sphere[0],
                self.sphere.to_tangent(
                    gs.random.normal(size=self.sphere.shape), self.y_sphere[0]
                ),
            ]
        )

        # Set up for special euclidean
        self.se2 = SpecialEuclidean(n=2)

        self.intercept_se2_true = self.se2.random_point()
        self.coef_se2_true = self.se2.to_tangent(
            5.0 * gs.random.rand(*self.se2.shape), self.intercept_se2_true
        )

        self.param_se2_true = gs.vstack(
            [
                gs.flatten(self.intercept_se2_true),
                gs.flatten(self.coef_se2_true),
            ]
        )

        X = gs.random.rand(self.n_samples)
        self.X_se2 = X - gs.mean(X)
        self.y_se2 = self.se2.metric.exp(
            self.X_se2[:, None, None] * self.coef_se2_true[None],
            self.intercept_se2_true,
        )

        self.param_se2_guess = gs.vstack(
            [
                gs.flatten(self.y_se2[0]),
                gs.flatten(
                    self.se2.to_tangent(
                        gs.random.normal(size=self.se2.shape), self.y_se2[0]
                    )
                ),
            ]
        )

        # Set up for discrete curves
        k_sampling_points = 8
        self.curves_2d = DiscreteCurves(
            Euclidean(dim=2), k_sampling_points=k_sampling_points
        )

        X = gs.random.rand(self.n_samples)
        self.X_curves_2d = X - gs.mean(X)

        self.intercept_curves_2d_true = self.curves_2d.random_point()
        self.coef_curves_2d_true = self.curves_2d.to_tangent(
            5.0 * gs.random.rand(*self.curves_2d.shape), self.intercept_curves_2d_true
        )

        # Added because of GitHub issue #1575
        intercept_curves_2d_true_repeated = gs.tile(
            gs.expand_dims(self.intercept_curves_2d_true, axis=0),
            (self.n_samples, 1, 1),
        )
        self.y_curves_2d = self.curves_2d.metric.exp(
            self.X_curves_2d[:, None, None] * self.coef_curves_2d_true[None],
            intercept_curves_2d_true_repeated,
        )

        self.param_curves_2d_true = gs.vstack(
            [
                gs.flatten(self.intercept_curves_2d_true),
                gs.flatten(self.coef_curves_2d_true),
            ]
        )
        self.param_curves_2d_guess = gs.vstack(
            [
                gs.flatten(self.y_curves_2d[0]),
                gs.flatten(
                    self.curves_2d.to_tangent(
                        gs.random.normal(size=self.curves_2d.shape), self.y_curves_2d[0]
                    )
                ),
            ]
        )

    def test_loss_euclidean(self):
        """Test that the loss is 0 at the true parameters."""
        gr = GeodesicRegression(
            self.eucl,
            center_X=False,
            method="extrinsic",
        )
        loss = gr._loss(
            self.X_eucl,
            self.y_eucl,
            self.param_eucl_true,
        )
        self.assertAllClose(loss.shape, ())
        self.assertTrue(gs.isclose(loss, 0.0))

    def test_loss_hypersphere(self):
        """Test that the loss is 0 at the true parameters."""
        gr = GeodesicRegression(
            self.sphere,
            center_X=False,
            method="extrinsic",
        )
        loss = gr._loss(
            self.X_sphere,
            self.y_sphere,
            self.param_sphere_true,
        )
        self.assertAllClose(loss.shape, ())
        self.assertTrue(gs.isclose(loss, 0.0))

    def test_loss_se2(self):
        """Test that the loss is 0 at the true parameters."""
        gr = GeodesicRegression(
            self.se2,
            center_X=False,
            method="extrinsic",
        )
        loss = gr._loss(self.X_se2, self.y_se2, self.param_se2_true)
        self.assertAllClose(loss.shape, ())
        self.assertTrue(gs.isclose(loss, 0.0))

    def test_loss_curves_2d(self):
        """Test that the loss is 0 at the true parameters."""
        gr = GeodesicRegression(
            self.curves_2d,
            center_X=False,
            method="extrinsic",
        )
        loss = gr._loss(
            self.X_curves_2d,
            self.y_curves_2d,
            self.param_curves_2d_true,
        )
        self.assertAllClose(loss.shape, ())
        self.assertTrue(gs.isclose(loss, 0.0))

    @tests.conftest.autograd_and_torch_only
    def test_value_and_grad_loss_euclidean(self):
        gr = GeodesicRegression(
            self.eucl,
            center_X=False,
            method="extrinsic",
            regularization=0.0,
        )

        def loss_of_param(param):
            return gr._loss(self.X_eucl, self.y_eucl, param)

        # Without numpy conversion
        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param)
        loss_value, loss_grad = objective_with_grad(self.param_eucl_guess)

        expected_grad_shape = (2, self.eucl.dim)
        self.assertAllClose(loss_value.shape, ())
        self.assertAllClose(loss_grad.shape, expected_grad_shape)

        self.assertFalse(gs.isclose(loss_value, 0.0))
        self.assertFalse(gs.isnan(loss_value))
        self.assertFalse(gs.all(gs.isclose(loss_grad, gs.zeros(expected_grad_shape))))
        self.assertTrue(gs.all(~gs.isnan(loss_grad)))

        # With numpy conversion
        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param, to_numpy=True)
        loss_value, loss_grad = objective_with_grad(self.param_eucl_guess)
        # Convert back to arrays/tensors
        loss_value = gs.array(loss_value)
        loss_grad = gs.array(loss_grad)

        expected_grad_shape = (2, self.eucl.dim)
        self.assertAllClose(loss_value.shape, ())
        self.assertAllClose(loss_grad.shape, expected_grad_shape)

        self.assertFalse(gs.isclose(loss_value, 0.0))
        self.assertFalse(gs.isnan(loss_value))
        self.assertFalse(gs.all(gs.isclose(loss_grad, gs.zeros(expected_grad_shape))))
        self.assertTrue(gs.all(~gs.isnan(loss_grad)))

    @tests.conftest.autograd_and_torch_only
    def test_value_and_grad_loss_hypersphere(self):
        gr = GeodesicRegression(
            self.sphere,
            center_X=False,
            method="extrinsic",
        )

        def loss_of_param(param):
            return gr._loss(self.X_sphere, self.y_sphere, param)

        # Without numpy conversion
        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param)
        loss_value, loss_grad = objective_with_grad(self.param_sphere_guess)

        expected_grad_shape = (2, self.sphere.dim + 1)
        self.assertAllClose(loss_value.shape, ())
        self.assertAllClose(loss_grad.shape, expected_grad_shape)

        self.assertFalse(gs.isclose(loss_value, 0.0))
        self.assertFalse(gs.isnan(loss_value))
        self.assertFalse(gs.all(gs.isclose(loss_grad, gs.zeros(expected_grad_shape))))
        self.assertTrue(gs.all(~gs.isnan(loss_grad)))

        # With numpy conversion
        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param, to_numpy=True)
        loss_value, loss_grad = objective_with_grad(self.param_sphere_guess)
        # Convert back to arrays/tensors
        loss_value = gs.array(loss_value)
        loss_grad = gs.array(loss_grad)

        expected_grad_shape = (2, self.sphere.dim + 1)
        self.assertAllClose(loss_value.shape, ())
        self.assertAllClose(loss_grad.shape, expected_grad_shape)

        self.assertFalse(gs.isclose(loss_value, 0.0))
        self.assertFalse(gs.isnan(loss_value))
        self.assertFalse(gs.all(gs.isclose(loss_grad, gs.zeros(expected_grad_shape))))
        self.assertTrue(gs.all(~gs.isnan(loss_grad)))

    @tests.conftest.autograd_only
    def test_value_and_grad_loss_se2(self):
        gr = GeodesicRegression(
            self.se2,
            center_X=False,
            method="extrinsic",
        )

        def loss_of_param(param):
            return gr._loss(self.X_se2, self.y_se2, param)

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param, to_numpy=True)
        loss_value, loss_grad = objective_with_grad(self.param_se2_true)
        expected_grad_shape = (
            2,
            math.prod(self.se2.shape),
        )

        self.assertTrue(gs.isclose(loss_value, 0.0))

        loss_value, loss_grad = objective_with_grad(self.param_se2_guess)

        self.assertAllClose(loss_value.shape, ())
        self.assertAllClose(loss_grad.shape, expected_grad_shape)

        self.assertFalse(gs.isclose(loss_value, 0.0))
        self.assertFalse(gs.all(gs.isclose(loss_grad, gs.zeros(expected_grad_shape))))
        self.assertTrue(gs.all(~gs.isnan(loss_grad)))

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param, to_numpy=True)
        loss_value, loss_grad = objective_with_grad(self.param_se2_guess)
        # TODO: fix autodiff to output proper type
        loss_value, loss_grad = gs.array(loss_value), gs.array(loss_grad)
        expected_grad_shape = (
            2,
            math.prod(self.se2.shape),
        )
        self.assertAllClose(loss_value.shape, ())
        self.assertAllClose(loss_grad.shape, expected_grad_shape)

        self.assertFalse(gs.isclose(loss_value, 0.0))
        self.assertFalse(gs.isnan(loss_value))
        self.assertFalse(gs.all(gs.isclose(loss_grad, gs.zeros(expected_grad_shape))))
        self.assertTrue(gs.all(~gs.isnan(loss_grad)))

    @tests.conftest.autograd_and_torch_only
    def test_loss_minimization_extrinsic_euclidean(self):
        """Minimize loss from noiseless data."""
        gr = GeodesicRegression(self.eucl, regularization=0.0)

        def loss_of_param(param):
            return gr._loss(self.X_eucl, self.y_eucl, param)

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param, to_numpy=True)
        initial_guess = gs.flatten(self.param_eucl_guess)
        res = minimize(
            objective_with_grad,
            initial_guess,
            method="CG",
            jac=True,
            tol=10 * gs.atol,
            options={"disp": True, "maxiter": 50},
        )
        self.assertAllClose(gs.array(res.x).shape, (self.eucl.dim * 2,))
        self.assertAllClose(res.fun, 0.0, atol=1000 * gs.atol)

        # Cast required because minimization happens in scipy in float64
        param_hat = gs.cast(gs.array(res.x), self.param_eucl_true.dtype)

        intercept_hat, coef_hat = gs.split(param_hat, 2)
        coef_hat = self.eucl.to_tangent(coef_hat, intercept_hat)
        self.assertAllClose(intercept_hat, self.intercept_eucl_true)

        tangent_vec_of_transport = self.eucl.metric.log(
            self.intercept_eucl_true, base_point=intercept_hat
        )

        transported_coef_hat = self.eucl.metric.parallel_transport(
            tangent_vec=coef_hat,
            base_point=intercept_hat,
            direction=tangent_vec_of_transport,
        )

        self.assertAllClose(
            transported_coef_hat, self.coef_eucl_true, atol=10 * gs.atol
        )

    @tests.conftest.autograd_and_torch_only
    def test_loss_minimization_extrinsic_hypersphere(self):
        """Minimize loss from noiseless data."""
        gr = GeodesicRegression(self.sphere, regularization=0.0)

        def loss_of_param(param):
            return gr._loss(self.X_sphere, self.y_sphere, param)

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param, to_numpy=True)
        initial_guess = gs.flatten(self.param_sphere_guess)
        res = minimize(
            objective_with_grad,
            initial_guess,
            method="CG",
            jac=True,
            tol=10 * gs.atol,
            options={"disp": True, "maxiter": 50},
        )
        self.assertAllClose(gs.array(res.x).shape, ((self.sphere.dim + 1) * 2,))
        self.assertAllClose(res.fun, 0.0, atol=5e-3)

        # Cast required because minimization happens in scipy in float64
        param_hat = gs.cast(gs.array(res.x), self.param_sphere_true.dtype)

        intercept_hat, coef_hat = gs.split(param_hat, 2)
        intercept_hat = self.sphere.projection(intercept_hat)
        coef_hat = self.sphere.to_tangent(coef_hat, intercept_hat)
        self.assertAllClose(intercept_hat, self.intercept_sphere_true, atol=5e-2)

        tangent_vec_of_transport = self.sphere.metric.log(
            self.intercept_sphere_true, base_point=intercept_hat
        )

        transported_coef_hat = self.sphere.metric.parallel_transport(
            tangent_vec=coef_hat,
            base_point=intercept_hat,
            direction=tangent_vec_of_transport,
        )

        self.assertAllClose(transported_coef_hat, self.coef_sphere_true, atol=0.6)

    @tests.conftest.autograd_only
    def test_loss_minimization_extrinsic_se2(self):
        gr = GeodesicRegression(
            self.se2,
            center_X=False,
            method="extrinsic",
        )

        def loss_of_param(param):
            return gr._loss(self.X_se2, self.y_se2, param)

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param, to_numpy=True)

        res = minimize(
            objective_with_grad,
            gs.flatten(self.param_se2_guess),
            method="CG",
            jac=True,
            options={"disp": True, "maxiter": 50},
        )
        self.assertAllClose(gs.array(res.x).shape, (18,))

        self.assertAllClose(res.fun, 0.0, atol=1e-6)

        # Cast required because minimization happens in scipy in float64
        param_hat = gs.cast(gs.array(res.x), self.param_se2_true.dtype)

        intercept_hat, coef_hat = gs.split(param_hat, 2)
        intercept_hat = gs.reshape(intercept_hat, self.se2.shape)
        coef_hat = gs.reshape(coef_hat, self.se2.shape)

        intercept_hat = self.se2.projection(intercept_hat)
        coef_hat = self.se2.to_tangent(coef_hat, intercept_hat)
        self.assertAllClose(intercept_hat, self.intercept_se2_true, atol=1e-4)

        tangent_vec_of_transport = self.se2.metric.log(
            self.intercept_se2_true, base_point=intercept_hat
        )

        transported_coef_hat = self.se2.metric.parallel_transport(
            tangent_vec=coef_hat,
            base_point=intercept_hat,
            direction=tangent_vec_of_transport,
        )

        self.assertAllClose(transported_coef_hat, self.coef_se2_true, atol=0.6)

    @tests.conftest.autograd_and_torch_only
    def test_fit_extrinsic_euclidean(self):
        gr = GeodesicRegression(
            self.eucl,
            center_X=False,
            method="extrinsic",
            initialization="random",
            regularization=0.9,
            compute_training_score=True,
        )
        gr.optimizer.options["maxiter"] = 50

        gr.fit(self.X_eucl, self.y_eucl)

        training_score = gr.training_score_
        intercept_hat, coef_hat = gr.intercept_, gr.coef_
        self.assertAllClose(intercept_hat.shape, self.eucl.shape)
        self.assertAllClose(coef_hat.shape, self.eucl.shape)
        self.assertAllClose(training_score, 1.0, atol=500 * gs.atol)
        self.assertAllClose(intercept_hat, self.intercept_eucl_true)

        tangent_vec_of_transport = self.eucl.metric.log(
            self.intercept_eucl_true, base_point=intercept_hat
        )

        transported_coef_hat = self.eucl.metric.parallel_transport(
            tangent_vec=coef_hat,
            base_point=intercept_hat,
            direction=tangent_vec_of_transport,
        )

        self.assertAllClose(transported_coef_hat, self.coef_eucl_true)

    @tests.conftest.autograd_and_torch_only
    def test_fit_extrinsic_hypersphere(self):
        gr = GeodesicRegression(
            self.sphere,
            center_X=False,
            method="extrinsic",
            initialization="random",
            regularization=0.9,
            compute_training_score=True,
        )
        gr.optimizer.options["maxiter"] = 50

        gr.fit(self.X_sphere, self.y_sphere)

        training_score = gr.training_score_
        intercept_hat, coef_hat = gr.intercept_, gr.coef_
        self.assertAllClose(intercept_hat.shape, self.sphere.shape)
        self.assertAllClose(coef_hat.shape, self.sphere.shape)
        self.assertAllClose(training_score, 1.0, atol=500 * gs.atol)
        self.assertAllClose(intercept_hat, self.intercept_sphere_true, atol=5e-3)

        tangent_vec_of_transport = self.sphere.metric.log(
            self.intercept_sphere_true, base_point=intercept_hat
        )

        transported_coef_hat = self.sphere.metric.parallel_transport(
            tangent_vec=coef_hat,
            base_point=intercept_hat,
            direction=tangent_vec_of_transport,
        )

        self.assertAllClose(transported_coef_hat, self.coef_sphere_true, atol=0.6)

    @tests.conftest.autograd_only
    def test_fit_extrinsic_se2(self):
        gr = GeodesicRegression(
            self.se2,
            center_X=False,
            method="extrinsic",
            initialization="warm_start",
            compute_training_score=True,
        )
        gr.optimizer.options["maxiter"] = 50

        gr.fit(self.X_se2, self.y_se2)
        intercept_hat, coef_hat = gr.intercept_, gr.coef_
        training_score = gr.training_score_

        self.assertAllClose(intercept_hat.shape, self.se2.shape)
        self.assertAllClose(coef_hat.shape, self.se2.shape)
        self.assertTrue(gs.isclose(training_score, 1.0))
        self.assertAllClose(intercept_hat, self.intercept_se2_true, atol=1e-4)

        tangent_vec_of_transport = self.se2.metric.log(
            self.intercept_se2_true, base_point=intercept_hat
        )

        transported_coef_hat = self.se2.metric.parallel_transport(
            tangent_vec=coef_hat,
            base_point=intercept_hat,
            direction=tangent_vec_of_transport,
        )

        self.assertAllClose(transported_coef_hat, self.coef_se2_true, atol=0.6)

    @tests.conftest.autograd_and_torch_only
    def test_fit_riemannian_euclidean(self):
        gr = GeodesicRegression(
            self.eucl,
            center_X=False,
            method="riemannian",
            compute_training_score=True,
        ).set(max_iter=50)

        gr.fit(self.X_eucl, self.y_eucl)
        intercept_hat, coef_hat = gr.intercept_, gr.coef_
        training_score = gr.training_score_

        self.assertAllClose(intercept_hat.shape, self.eucl.shape)
        self.assertAllClose(coef_hat.shape, self.eucl.shape)

        self.assertAllClose(training_score, 1.0, atol=0.1)
        self.assertAllClose(intercept_hat, self.intercept_eucl_true)

        tangent_vec_of_transport = self.eucl.metric.log(
            self.intercept_eucl_true, base_point=intercept_hat
        )

        transported_coef_hat = self.eucl.metric.parallel_transport(
            tangent_vec=coef_hat,
            base_point=intercept_hat,
            direction=tangent_vec_of_transport,
        )

        self.assertAllClose(transported_coef_hat, self.coef_eucl_true, atol=1e-2)

    @tests.conftest.autograd_and_torch_only
    def test_fit_riemannian_hypersphere(self):
        gr = GeodesicRegression(
            self.sphere,
            center_X=False,
            method="riemannian",
            compute_training_score=True,
        ).set(max_iter=50)

        gr.fit(self.X_sphere, self.y_sphere)
        intercept_hat, coef_hat = gr.intercept_, gr.coef_
        training_score = gr.training_score_

        self.assertAllClose(intercept_hat.shape, self.sphere.shape)
        self.assertAllClose(coef_hat.shape, self.sphere.shape)

        self.assertAllClose(training_score, 1.0, atol=0.1)
        self.assertAllClose(intercept_hat, self.intercept_sphere_true, atol=1e-2)

        tangent_vec_of_transport = self.sphere.metric.log(
            self.intercept_sphere_true, base_point=intercept_hat
        )

        transported_coef_hat = self.sphere.metric.parallel_transport(
            tangent_vec=coef_hat,
            base_point=intercept_hat,
            direction=tangent_vec_of_transport,
        )

        self.assertAllClose(transported_coef_hat, self.coef_sphere_true, atol=0.6)

    @tests.conftest.autograd_only
    def test_fit_riemannian_se2(self):
        init = (self.y_se2[0], gs.zeros_like(self.y_se2[0]))
        gr = GeodesicRegression(
            self.se2,
            center_X=False,
            method="riemannian",
            initialization=init,
            compute_training_score=True,
        ).set(max_iter=50)

        gr.fit(self.X_se2, self.y_se2)
        intercept_hat, coef_hat = gr.intercept_, gr.coef_
        training_score = gr.training_score_

        self.assertAllClose(intercept_hat.shape, self.se2.shape)
        self.assertAllClose(coef_hat.shape, self.se2.shape)
        self.assertAllClose(training_score, 1.0, atol=1e-4)
        self.assertAllClose(intercept_hat, self.intercept_se2_true, atol=1e-4)

        tangent_vec_of_transport = self.se2.metric.log(
            self.intercept_se2_true, base_point=intercept_hat
        )

        transported_coef_hat = self.se2.metric.parallel_transport(
            tangent_vec=coef_hat,
            base_point=intercept_hat,
            direction=tangent_vec_of_transport,
        )

        self.assertAllClose(transported_coef_hat, self.coef_se2_true, atol=0.6)
