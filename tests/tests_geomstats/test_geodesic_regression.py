"""Unit tests for Geodesic Regression."""

from scipy.optimize import minimize

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hypersphere import Hypersphere
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
        X = gs.random.rand(self.n_samples)
        self.X_sphere = X - gs.mean(X)
        self.intercept_sphere_true = self.sphere.random_point()
        self.coef_sphere_true = self.sphere.projection(
            gs.random.rand(self.dim_sphere + 1)
        )

        self.y_sphere = self.sphere.metric.exp(
            self.X_sphere[:, None] * self.coef_sphere_true,
            base_point=self.intercept_sphere_true,
        )

        self.param_sphere_true = gs.vstack(
            [self.intercept_sphere_true, self.coef_sphere_true]
        )
        self.param_sphere_guess = gs.vstack(
            [
                self.y_sphere[0],
                self.sphere.to_tangent(
                    gs.random.normal(size=self.shape_sphere), self.y_sphere[0]
                ),
            ]
        )

    def test_loss_hypersphere(self):
        gr = GeodesicRegression(
            self.sphere,
            metric=self.sphere.metric,
            center_X=False,
            method="extrinsic",
            verbose=True,
            max_iter=50,
            learning_rate=0.1,
        )
        loss = gr._loss(
            self.X_sphere,
            self.y_sphere,
            self.param_sphere_true,
            self.shape_sphere,
        )
        self.assertAllClose(loss.shape, ())
        self.assertTrue(gs.isclose(loss, 0.0))

    @geomstats.tests.autograd_tf_and_torch_only
    def test_value_and_grad_loss_hypersphere(self):
        gr = GeodesicRegression(
            self.sphere,
            metric=self.sphere.metric,
            center_X=False,
            method="extrinsic",
            verbose=True,
            max_iter=50,
            learning_rate=0.1,
        )

        def loss_of_param(param):
            return gr._loss(self.X_sphere, self.y_sphere, param, self.shape_sphere)

        # Without numpy conversion
        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param)
        loss_value, loss_grad = objective_with_grad(self.param_sphere_guess)

        expected_grad_shape = (2, self.dim_sphere + 1)
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

        expected_grad_shape = (2, self.dim_sphere + 1)
        self.assertAllClose(loss_value.shape, ())
        self.assertAllClose(loss_grad.shape, expected_grad_shape)

        self.assertFalse(gs.isclose(loss_value, 0.0))
        self.assertFalse(gs.isnan(loss_value))
        self.assertFalse(gs.all(gs.isclose(loss_grad, gs.zeros(expected_grad_shape))))
        self.assertTrue(gs.all(~gs.isnan(loss_grad)))

    @geomstats.tests.autograd_tf_and_torch_only
    def test_loss_minimization_extrinsic_hypersphere(self):
        """Minimize loss from noiseless data."""
        gr = GeodesicRegression(
            self.sphere,
            metric=self.sphere.metric,
            center_X=False,
            method="extrinsic",
            verbose=True,
            max_iter=50,
            learning_rate=0.1,
        )

        def loss_of_param(param):
            return gr._loss(self.X_sphere, self.y_sphere, param, self.shape_sphere)

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param, to_numpy=True)
        initial_guess = gs.flatten(self.param_sphere_guess)
        res = minimize(
            objective_with_grad,
            initial_guess,
            method="CG",
            jac=True,
            options={"disp": True, "maxiter": 50},
        )
        self.assertAllClose(gs.array(res.x).shape, ((self.dim_sphere + 1) * 2,))
        self.assertTrue(gs.isclose(res.fun, 0.0, atol=100 * gs.atol))

        # Cast required because minimization happens in scipy in float64
        param_hat = gs.cast(gs.array(res.x), self.param_sphere_true.dtype)

        intercept_hat, coef_hat = gs.split(param_hat, 2)
        intercept_hat = self.sphere.projection(intercept_hat)
        coef_hat = self.sphere.to_tangent(coef_hat, intercept_hat)
        self.assertAllClose(intercept_hat, self.intercept_sphere_true, atol=1e-3)

        tangent_vec_of_transport = self.sphere.metric.log(
            self.intercept_sphere_true, base_point=intercept_hat
        )

        transported_coef_hat = self.sphere.metric.parallel_transport(
            tangent_vec_a=coef_hat,
            tangent_vec_b=tangent_vec_of_transport,
            base_point=intercept_hat,
        )

        self.assertAllClose(transported_coef_hat, self.coef_sphere_true, atol=0.6)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_fit_extrinsic_hypersphere(self):
        gr = GeodesicRegression(
            self.sphere,
            metric=self.sphere.metric,
            center_X=False,
            method="extrinsic",
            verbose=True,
            max_iter=50,
            learning_rate=0.1,
        )

        gr.fit(self.X_sphere, self.y_sphere, compute_training_score=True)

        training_score = gr.training_score_
        intercept_hat, coef_hat = gr.intercept_, gr.coef_
        self.assertAllClose(intercept_hat.shape, self.shape_sphere)
        self.assertAllClose(coef_hat.shape, self.shape_sphere)
        self.assertAllClose(training_score, 1.0, atol=500 * gs.atol)
        self.assertAllClose(intercept_hat, self.intercept_sphere_true, atol=5e-3)

        tangent_vec_of_transport = self.sphere.metric.log(
            self.intercept_sphere_true, base_point=intercept_hat
        )

        transported_coef_hat = self.sphere.metric.parallel_transport(
            tangent_vec_a=coef_hat,
            tangent_vec_b=tangent_vec_of_transport,
            base_point=intercept_hat,
        )

        self.assertAllClose(transported_coef_hat, self.coef_sphere_true, atol=0.6)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_fit_riemannian_hypersphere(self):
        gr = GeodesicRegression(
            self.sphere,
            metric=self.sphere.metric,
            center_X=False,
            method="riemannian",
            verbose=True,
            max_iter=50,
            learning_rate=0.1,
        )

        gr.fit(self.X_sphere, self.y_sphere, compute_training_score=True)
        intercept_hat, coef_hat = gr.intercept_, gr.coef_
        training_score = gr.training_score_

        self.assertAllClose(intercept_hat.shape, self.shape_sphere)
        self.assertAllClose(coef_hat.shape, self.shape_sphere)

        self.assertAllClose(training_score, 1.0, atol=0.1)
        self.assertAllClose(intercept_hat, self.intercept_sphere_true, atol=5e-3)

        tangent_vec_of_transport = self.sphere.metric.log(
            self.intercept_sphere_true, base_point=intercept_hat
        )

        transported_coef_hat = self.sphere.metric.parallel_transport(
            tangent_vec_a=coef_hat,
            tangent_vec_b=tangent_vec_of_transport,
            base_point=intercept_hat,
        )

        self.assertAllClose(transported_coef_hat, self.coef_sphere_true, atol=0.6)
