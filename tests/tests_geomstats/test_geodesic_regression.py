"""Unit tests for Geodesic Regression."""

from scipy.optimize import minimize

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.learning.geodesic_regression import GeodesicRegression


class TestGeodesicRegression(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setup_method(self):
        gs.random.seed(1234)
        self.n_samples = 20

        # Set up for euclidean
        self.dim_eucl = 3
        self.shape_eucl = (self.dim_eucl,)
        self.eucl = Euclidean(dim=self.dim_eucl)
        X = gs.random.rand(self.n_samples)
        self.X_eucl = X - gs.mean(X)
        self.intercept_eucl_true = self.eucl.random_point()
        self.coef_eucl_true = self.eucl.random_point()

        self.y_eucl = (
            self.intercept_eucl_true + self.X_eucl[:, None] * self.coef_eucl_true
        )
        self.param_eucl_true = gs.vstack(
            [self.intercept_eucl_true, self.coef_eucl_true]
        )
        self.param_eucl_guess = gs.vstack(
            [self.y_eucl[0], self.y_eucl[0] + gs.random.normal(size=self.shape_eucl)]
        )

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

        # Set up for special euclidean
        self.se2 = SpecialEuclidean(n=2)
        self.metric_se2 = self.se2.left_canonical_metric
        self.metric_se2.default_point_type = "matrix"

        self.shape_se2 = (3, 3)
        X = gs.random.rand(self.n_samples)
        self.X_se2 = X - gs.mean(X)

        self.intercept_se2_true = self.se2.random_point()
        self.coef_se2_true = self.se2.to_tangent(
            5.0 * gs.random.rand(*self.shape_se2), self.intercept_se2_true
        )

        self.y_se2 = self.metric_se2.exp(
            self.X_se2[:, None, None] * self.coef_se2_true[None],
            self.intercept_se2_true,
        )

        self.param_se2_true = gs.vstack(
            [
                gs.flatten(self.intercept_se2_true),
                gs.flatten(self.coef_se2_true),
            ]
        )
        self.param_se2_guess = gs.vstack(
            [
                gs.flatten(self.y_se2[0]),
                gs.flatten(
                    self.se2.to_tangent(
                        gs.random.normal(size=self.shape_se2), self.y_se2[0]
                    )
                ),
            ]
        )

    def test_loss_euclidean(self):
        """Test that the loss is 0 at the true parameters."""
        gr = GeodesicRegression(
            self.eucl,
            metric=self.eucl.metric,
            center_X=False,
            method="extrinsic",
            max_iter=50,
            init_step_size=0.1,
            verbose=True,
        )
        loss = gr._loss(
            self.X_eucl,
            self.y_eucl,
            self.param_eucl_true,
            self.shape_eucl,
        )
        self.assertAllClose(loss.shape, ())
        self.assertTrue(gs.isclose(loss, 0.0))

    def test_loss_hypersphere(self):
        """Test that the loss is 0 at the true parameters."""
        gr = GeodesicRegression(
            self.sphere,
            metric=self.sphere.metric,
            center_X=False,
            method="extrinsic",
            max_iter=50,
            init_step_size=0.1,
            verbose=True,
        )
        loss = gr._loss(
            self.X_sphere,
            self.y_sphere,
            self.param_sphere_true,
            self.shape_sphere,
        )
        self.assertAllClose(loss.shape, ())
        self.assertTrue(gs.isclose(loss, 0.0))

    @geomstats.tests.autograd_and_tf_only
    def test_loss_se2(self):
        """Test that the loss is 0 at the true parameters."""
        gr = GeodesicRegression(
            self.se2,
            metric=self.metric_se2,
            center_X=False,
            method="extrinsic",
            max_iter=50,
            init_step_size=0.1,
            verbose=True,
        )
        loss = gr._loss(self.X_se2, self.y_se2, self.param_se2_true, self.shape_se2)
        self.assertAllClose(loss.shape, ())
        self.assertTrue(gs.isclose(loss, 0.0))

    @geomstats.tests.autograd_tf_and_torch_only
    def test_value_and_grad_loss_euclidean(self):
        gr = GeodesicRegression(
            self.eucl,
            metric=self.eucl.metric,
            center_X=False,
            method="extrinsic",
            max_iter=50,
            init_step_size=0.1,
            verbose=True,
            regularization=0,
        )

        def loss_of_param(param):
            return gr._loss(self.X_eucl, self.y_eucl, param, self.shape_eucl)

        # Without numpy conversion
        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param)
        loss_value, loss_grad = objective_with_grad(self.param_eucl_guess)

        expected_grad_shape = (2, self.dim_eucl)
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

        expected_grad_shape = (2, self.dim_eucl)
        self.assertAllClose(loss_value.shape, ())
        self.assertAllClose(loss_grad.shape, expected_grad_shape)

        self.assertFalse(gs.isclose(loss_value, 0.0))
        self.assertFalse(gs.isnan(loss_value))
        self.assertFalse(gs.all(gs.isclose(loss_grad, gs.zeros(expected_grad_shape))))
        self.assertTrue(gs.all(~gs.isnan(loss_grad)))

    @geomstats.tests.autograd_tf_and_torch_only
    def test_value_and_grad_loss_hypersphere(self):
        gr = GeodesicRegression(
            self.sphere,
            metric=self.sphere.metric,
            center_X=False,
            method="extrinsic",
            max_iter=50,
            init_step_size=0.1,
            verbose=True,
            regularization=0,
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

    @geomstats.tests.autograd_and_tf_only
    def test_value_and_grad_loss_se2(self):

        gr = GeodesicRegression(
            self.se2,
            metric=self.metric_se2,
            center_X=False,
            method="extrinsic",
            max_iter=50,
            init_step_size=0.1,
            verbose=True,
        )

        def loss_of_param(param):
            return gr._loss(self.X_se2, self.y_se2, param, self.shape_se2)

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param)
        loss_value, loss_grad = objective_with_grad(self.param_se2_true)
        expected_grad_shape = (
            2,
            self.shape_se2[0] * self.shape_se2[1],
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
        expected_grad_shape = (
            2,
            self.shape_se2[0] * self.shape_se2[1],
        )
        self.assertAllClose(loss_value.shape, ())
        self.assertAllClose(loss_grad.shape, expected_grad_shape)

        self.assertFalse(gs.isclose(loss_value, 0.0))
        self.assertFalse(gs.isnan(loss_value))
        self.assertFalse(gs.all(gs.isclose(loss_grad, gs.zeros(expected_grad_shape))))
        self.assertTrue(gs.all(~gs.isnan(loss_grad)))

    @geomstats.tests.autograd_tf_and_torch_only
    def test_loss_minimization_extrinsic_euclidean(self):
        """Minimize loss from noiseless data."""
        gr = GeodesicRegression(self.eucl, regularization=0)

        def loss_of_param(param):
            return gr._loss(self.X_eucl, self.y_eucl, param, self.shape_eucl)

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
        self.assertAllClose(gs.array(res.x).shape, (self.dim_eucl * 2,))
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

        self.assertAllClose(transported_coef_hat, self.coef_eucl_true)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_loss_minimization_extrinsic_hypersphere(self):
        """Minimize loss from noiseless data."""
        gr = GeodesicRegression(self.sphere, regularization=0)

        def loss_of_param(param):
            return gr._loss(self.X_sphere, self.y_sphere, param, self.shape_sphere)

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
        self.assertAllClose(gs.array(res.x).shape, ((self.dim_sphere + 1) * 2,))
        self.assertAllClose(res.fun, 0.0, atol=1000 * gs.atol)

        # Cast required because minimization happens in scipy in float64
        param_hat = gs.cast(gs.array(res.x), self.param_sphere_true.dtype)

        intercept_hat, coef_hat = gs.split(param_hat, 2)
        intercept_hat = self.sphere.projection(intercept_hat)
        coef_hat = self.sphere.to_tangent(coef_hat, intercept_hat)
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

    @geomstats.tests.autograd_and_tf_only
    def test_loss_minimization_extrinsic_se2(self):
        gr = GeodesicRegression(
            self.se2,
            metric=self.metric_se2,
            center_X=False,
            method="extrinsic",
            max_iter=50,
            init_step_size=0.1,
            verbose=True,
        )

        def loss_of_param(param):
            return gr._loss(self.X_se2, self.y_se2, param, self.shape_se2)

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param, to_numpy=True)

        res = minimize(
            objective_with_grad,
            gs.flatten(self.param_se2_guess),
            method="CG",
            jac=True,
            options={"disp": True, "maxiter": 50},
        )
        self.assertAllClose(gs.array(res.x).shape, (18,))

        self.assertTrue(gs.isclose(res.fun, 0.0))

        # Cast required because minimization happens in scipy in float64
        param_hat = gs.cast(gs.array(res.x), self.param_se2_true.dtype)

        intercept_hat, coef_hat = gs.split(param_hat, 2)
        intercept_hat = gs.reshape(intercept_hat, self.shape_se2)
        coef_hat = gs.reshape(coef_hat, self.shape_se2)

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

    @geomstats.tests.autograd_tf_and_torch_only
    def test_fit_extrinsic_euclidean(self):
        gr = GeodesicRegression(
            self.eucl,
            metric=self.eucl.metric,
            center_X=False,
            method="extrinsic",
            max_iter=50,
            init_step_size=0.1,
            verbose=True,
            initialization="random",
            regularization=0.9,
        )

        gr.fit(self.X_eucl, self.y_eucl, compute_training_score=True)

        training_score = gr.training_score_
        intercept_hat, coef_hat = gr.intercept_, gr.coef_
        self.assertAllClose(intercept_hat.shape, self.shape_eucl)
        self.assertAllClose(coef_hat.shape, self.shape_eucl)
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

    @geomstats.tests.autograd_tf_and_torch_only
    def test_fit_extrinsic_hypersphere(self):
        gr = GeodesicRegression(
            self.sphere,
            metric=self.sphere.metric,
            center_X=False,
            method="extrinsic",
            max_iter=50,
            init_step_size=0.1,
            verbose=True,
            initialization="random",
            regularization=0.9,
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
            tangent_vec=coef_hat,
            base_point=intercept_hat,
            direction=tangent_vec_of_transport,
        )

        self.assertAllClose(transported_coef_hat, self.coef_sphere_true, atol=0.6)

    @geomstats.tests.autograd_and_tf_only
    def test_fit_extrinsic_se2(self):
        gr = GeodesicRegression(
            self.se2,
            metric=self.metric_se2,
            center_X=False,
            method="extrinsic",
            max_iter=50,
            init_step_size=0.1,
            verbose=True,
            initialization="warm_start",
        )

        gr.fit(self.X_se2, self.y_se2, compute_training_score=True)
        intercept_hat, coef_hat = gr.intercept_, gr.coef_
        training_score = gr.training_score_

        self.assertAllClose(intercept_hat.shape, self.shape_se2)
        self.assertAllClose(coef_hat.shape, self.shape_se2)
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

    @geomstats.tests.autograd_tf_and_torch_only
    def test_fit_riemannian_euclidean(self):
        gr = GeodesicRegression(
            self.eucl,
            metric=self.eucl.metric,
            center_X=False,
            method="riemannian",
            max_iter=50,
            init_step_size=0.1,
            verbose=True,
        )

        gr.fit(self.X_eucl, self.y_eucl, compute_training_score=True)
        intercept_hat, coef_hat = gr.intercept_, gr.coef_
        training_score = gr.training_score_

        self.assertAllClose(intercept_hat.shape, self.shape_eucl)
        self.assertAllClose(coef_hat.shape, self.shape_eucl)

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

    @geomstats.tests.autograd_tf_and_torch_only
    def test_fit_riemannian_hypersphere(self):
        gr = GeodesicRegression(
            self.sphere,
            metric=self.sphere.metric,
            center_X=False,
            method="riemannian",
            max_iter=50,
            init_step_size=0.1,
            verbose=True,
        )

        gr.fit(self.X_sphere, self.y_sphere, compute_training_score=True)
        intercept_hat, coef_hat = gr.intercept_, gr.coef_
        training_score = gr.training_score_

        self.assertAllClose(intercept_hat.shape, self.shape_sphere)
        self.assertAllClose(coef_hat.shape, self.shape_sphere)

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

    @geomstats.tests.autograd_and_tf_only
    def test_fit_riemannian_se2(self):
        init = (self.y_se2[0], gs.zeros_like(self.y_se2[0]))
        gr = GeodesicRegression(
            self.se2,
            metric=self.metric_se2,
            center_X=False,
            method="riemannian",
            max_iter=50,
            init_step_size=0.1,
            verbose=True,
            initialization=init,
        )

        gr.fit(self.X_se2, self.y_se2, compute_training_score=True)
        intercept_hat, coef_hat = gr.intercept_, gr.coef_
        training_score = gr.training_score_

        self.assertAllClose(intercept_hat.shape, self.shape_se2)
        self.assertAllClose(coef_hat.shape, self.shape_se2)
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
