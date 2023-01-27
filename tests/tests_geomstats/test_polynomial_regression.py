"""Unit tests for Polynomial Regression."""
from scipy.optimize import minimize

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.discrete_curves import R2, DiscreteCurves
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.learning.polynomial_regression import PolynomialRegression


class TestPolynomialRegression(tests.conftest.TestCase):
    _multiprocess_can_split_ = True

    def setup_method(self):
        gs.random.seed(12345)
        self.n_samples = 20

        # Set up for euclidean
        self.order_eucl = 3

        self.dim_eucl = 3
        self.shape_eucl = (self.dim_eucl,)
        self.eucl = Euclidean(dim=self.dim_eucl)
        X = gs.random.rand(self.n_samples)
        self.X_eucl = X - gs.mean(X)
        self.intercept_eucl_true = self.eucl.random_point()
        # Needs to be order x dim shape
        self.coef_eucl_true = gs.random.rand(self.order_eucl, self.dim_eucl)
        # Make matrix of X to nth power by columns
        X_powers = gs.vstack([self.X_eucl**k for k in range(1, self.order_eucl + 1)])
        # Use matrix multiplication
        self.y_eucl = self.intercept_eucl_true + Matrices.mul(
            gs.transpose(X_powers), self.coef_eucl_true
        )

        self.param_eucl_true = gs.vstack(
            [self.intercept_eucl_true, self.coef_eucl_true]
        )
        self.param_eucl_guess = gs.vstack(
            [
                self.y_eucl[0],
                self.y_eucl[0]
                + gs.random.normal(size=(self.order_eucl,) + self.shape_eucl),
            ]
        )

        # Set up for hypersphere
        self.dim_sphere = 4
        self.order_sphere = 2
        self.shape_sphere = (self.dim_sphere + 1,)
        self.sphere = Hypersphere(dim=self.dim_sphere)
        X = gs.random.rand(self.n_samples)
        self.X_sphere = X - gs.mean(X)

        self.intercept_sphere_true = self.sphere.random_point()

        self.coef_sphere_true = self.sphere.projection(
            gs.random.rand(self.order_sphere, self.dim_sphere + 1)
        )

        # Make matrix of X to nth power by columns
        X_powers = gs.vstack(
            [self.X_sphere**k for k in range(1, self.order_sphere + 1)]
        )
        # Use matrix multiplication

        self.y_sphere = self.sphere.metric.exp(
            Matrices.mul(gs.transpose(X_powers), self.coef_sphere_true),
            base_point=self.intercept_sphere_true,
        )

        self.param_sphere_true = gs.vstack(
            [self.intercept_sphere_true, self.coef_sphere_true]
        )
        self.param_sphere_guess = gs.vstack(
            [
                self.y_sphere[0],
                self.sphere.to_tangent(
                    gs.random.normal(size=(self.order_sphere,) + self.shape_sphere),
                    self.y_sphere[0],
                ),
            ]
        )

        # Set up for special euclidean
        self.se2 = SpecialEuclidean(n=2)
        self.order_se2 = 2
        self.metric_se2 = self.se2.left_canonical_metric
        # self.metric_se2.default_point_type = "matrix"

        self.shape_se2 = (3, 3)
        X = gs.random.rand(self.n_samples)
        self.X_se2 = X - gs.mean(X)

        self.intercept_se2_true = self.se2.random_point()

        self.coef_se2_true = self.se2.to_tangent(
            5.0 * gs.random.rand(*((self.order_se2,) + self.shape_se2)),
            self.intercept_se2_true,
        )
        self.coef_se2_true = gs.squeeze(self.coef_se2_true)
        # Make matrix of X to nth power by columns
        X_powers = gs.vstack([self.X_se2**k for k in range(1, self.order_se2 + 1)])
        # Use matrix multiplication
        # Reshape twice to multiply with use multidimensonal array
        self.y_se2 = self.metric_se2.exp(
            gs.reshape(
                Matrices.mul(
                    gs.transpose(X_powers),
                    gs.reshape(self.coef_se2_true, (self.order_se2, -1)),
                ),
                (-1,) + self.shape_se2,
            ),
            self.intercept_se2_true,
        )
        self.param_se2_true = gs.vstack(
            [
                gs.flatten(self.intercept_se2_true),
                gs.reshape(self.coef_se2_true, (self.order_se2, -1)),
            ]
        )

        self.param_se2_guess = gs.vstack(
            [
                gs.flatten(self.y_se2[0]),
                gs.reshape(
                    self.se2.to_tangent(
                        gs.random.normal(size=(self.order_se2,) + self.shape_se2),
                        self.y_se2[0],
                    ),
                    (self.order_se2, -1),
                ),
            ]
        )

        # Set up for discrete curves
        n_sampling_points = 8
        self.curves_2d = DiscreteCurves(R2, k_sampling_points=n_sampling_points)
        self.order_curves_2d = 3
        self.metric_curves_2d = self.curves_2d.srv_metric

        self.shape_curves_2d = (n_sampling_points, 2)
        X = gs.random.rand(self.n_samples)
        self.X_curves_2d = X - gs.mean(X)

        self.intercept_curves_2d_true = self.curves_2d.random_point()
        self.coef_curves_2d_true = self.curves_2d.to_tangent(
            5.0 * gs.random.rand(*((self.order_curves_2d,) + self.shape_curves_2d)),
            self.intercept_curves_2d_true,
        )

        # Make matrix of X to nth power by columns
        X_powers = gs.vstack(
            [self.X_curves_2d**k for k in range(1, self.order_curves_2d + 1)]
        )
        # Use matrix multiplication #reshape twice to use multidimensonal array

        # Added because of GitHub issue #1575
        intercept_curves_2d_true_repeated = gs.tile(
            gs.expand_dims(self.intercept_curves_2d_true, axis=0),
            (self.n_samples, 1, 1),
        )
        self.y_curves_2d = self.metric_curves_2d.exp(
            gs.reshape(
                Matrices.mul(
                    gs.transpose(X_powers),
                    gs.reshape(self.coef_curves_2d_true, (self.order_curves_2d, -1)),
                ),
                (-1,) + self.shape_curves_2d,
            ),
            # self.X_curves_2d[:, None, None] * self.coef_curves_2d_true[None],
            intercept_curves_2d_true_repeated,
        )

        self.param_curves_2d_true = gs.vstack(
            [
                gs.flatten(self.intercept_curves_2d_true),
                gs.reshape(self.coef_curves_2d_true, (self.order_curves_2d, -1)),
            ]
        )
        self.param_curves_2d_guess = gs.vstack(
            [
                gs.flatten(self.y_curves_2d[0]),
                gs.reshape(
                    self.curves_2d.to_tangent(
                        gs.random.normal(
                            size=(self.order_curves_2d,) + self.shape_curves_2d
                        ),
                        self.y_curves_2d[0],
                    ),
                    (self.order_curves_2d, -1),
                ),
            ]
        )

    def test_loss_euclidean(self):
        """Test that the loss is 0 at the true parameters."""
        pr = PolynomialRegression(
            self.eucl,
            metric=self.eucl.metric,
            order=self.order_eucl,
            center_X=False,
            method="extrinsic",
            max_iter=50,
            init_step_size=0.1,
            verbose=True,
        )
        loss = pr._loss(
            self.X_eucl,
            self.y_eucl,
            self.param_eucl_true,
            self.shape_eucl,
        )

        self.assertAllClose(loss.shape, ())
        self.assertTrue(gs.isclose(loss, 0.0))

    def test_loss_hypersphere(self):
        """Test that the loss is 0 at the true parameters."""
        pr = PolynomialRegression(
            self.sphere,
            metric=self.sphere.metric,
            order=self.order_sphere,
            center_X=False,
            method="extrinsic",
            max_iter=50,
            init_step_size=0.1,
            verbose=True,
        )
        loss = pr._loss(
            self.X_sphere,
            self.y_sphere,
            self.param_sphere_true,
            self.shape_sphere,
        )
        self.assertAllClose(loss.shape, ())
        self.assertTrue(gs.isclose(loss, 0.0))

    @tests.conftest.autograd_only
    def test_loss_se2(self):
        """Test that the loss is 0 at the true parameters."""
        pr = PolynomialRegression(
            self.se2,
            metric=self.metric_se2,
            order=self.order_se2,
            center_X=False,
            method="extrinsic",
            max_iter=50,
            init_step_size=0.1,
            verbose=True,
        )
        loss = pr._loss(self.X_se2, self.y_se2, self.param_se2_true, self.shape_se2)
        self.assertAllClose(loss.shape, ())
        self.assertTrue(gs.isclose(loss, 0.0))

    @tests.conftest.autograd_only
    def test_loss_curves_2d(self):
        """Test that the loss is 0 at the true parameters."""
        pr = PolynomialRegression(
            self.curves_2d,
            metric=self.metric_curves_2d,
            order=self.order_curves_2d,
            center_X=False,
            method="extrinsic",
            max_iter=50,
            init_step_size=0.1,
            verbose=True,
        )
        loss = pr._loss(
            self.X_curves_2d,
            self.y_curves_2d,
            self.param_curves_2d_true,
            self.shape_curves_2d,
        )
        self.assertAllClose(loss.shape, ())
        self.assertTrue(gs.isclose(loss, 0.0))

    @tests.conftest.autograd_and_torch_only
    def test_value_and_grad_loss_euclidean(self):
        pr = PolynomialRegression(
            self.eucl,
            metric=self.eucl.metric,
            order=self.order_eucl,
            center_X=False,
            method="extrinsic",
            max_iter=50,
            init_step_size=0.1,
            verbose=True,
            regularization=0,
        )

        def loss_of_param(param):
            return pr._loss(self.X_eucl, self.y_eucl, param, self.shape_eucl)

        # Parameter/grad shape will be (order + 1, shape)
        # Compare with geodesic regression where order=1 -> (2, shape)
        expected_grad_shape = (self.order_eucl + 1,) + self.shape_eucl

        # Without numpy conversion
        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param)
        loss_value, loss_grad = objective_with_grad(self.param_eucl_guess)

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

        self.assertAllClose(loss_value.shape, ())
        self.assertAllClose(loss_grad.shape, expected_grad_shape)

        self.assertFalse(gs.isclose(loss_value, 0.0))
        self.assertFalse(gs.isnan(loss_value))
        self.assertFalse(gs.all(gs.isclose(loss_grad, gs.zeros(expected_grad_shape))))
        self.assertTrue(gs.all(~gs.isnan(loss_grad)))

    @tests.conftest.autograd_and_torch_only
    def test_value_and_grad_loss_hypersphere(self):
        pr = PolynomialRegression(
            self.sphere,
            metric=self.sphere.metric,
            order=self.order_sphere,
            center_X=False,
            method="extrinsic",
            max_iter=50,
            init_step_size=0.1,
            verbose=True,
            regularization=0,
        )

        def loss_of_param(param):
            return pr._loss(self.X_sphere, self.y_sphere, param, self.shape_sphere)

        # Parameter/grad shape will be (order + 1, shape - flattened)
        # Compare with geodesic regression where order=1 -> (2, shape - flattened)
        print(self.shape_sphere)
        expected_grad_shape = (self.order_sphere + 1,) + self.shape_sphere

        # Without numpy conversion
        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param)
        loss_value, loss_grad = objective_with_grad(self.param_sphere_guess)

        # print(f"Expected grad shape is: {expected_grad_shape}")
        # print(f"Loss grad shape is: {loss_grad.shape}")

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

        self.assertAllClose(loss_value.shape, ())
        self.assertAllClose(loss_grad.shape, expected_grad_shape)

        self.assertFalse(gs.isclose(loss_value, 0.0))
        self.assertFalse(gs.isnan(loss_value))
        self.assertFalse(gs.all(gs.isclose(loss_grad, gs.zeros(expected_grad_shape))))
        self.assertTrue(gs.all(~gs.isnan(loss_grad)))

    @tests.conftest.autograd_only
    def test_value_and_grad_loss_se2(self):

        pr = PolynomialRegression(
            self.se2,
            metric=self.metric_se2,
            order=self.order_se2,
            center_X=False,
            method="extrinsic",
            max_iter=50,
            init_step_size=0.1,
            verbose=True,
        )

        def loss_of_param(param):
            return pr._loss(self.X_se2, self.y_se2, param, self.shape_se2)

        expected_grad_shape = (self.order_se2 + 1, gs.prod(self.shape_se2))

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param)
        loss_value, loss_grad = objective_with_grad(self.param_se2_true)

        self.assertTrue(gs.isclose(loss_value, 0.0))

        loss_value, loss_grad = objective_with_grad(self.param_se2_guess)

        self.assertAllClose(loss_value.shape, ())
        self.assertAllClose(loss_grad.shape, expected_grad_shape)

        self.assertFalse(gs.isclose(loss_value, 0.0))
        self.assertFalse(gs.all(gs.isclose(loss_grad, gs.zeros(expected_grad_shape))))
        self.assertTrue(gs.all(~gs.isnan(loss_grad)))

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param, to_numpy=True)
        loss_value, loss_grad = objective_with_grad(self.param_se2_guess)
        self.assertAllClose(loss_value.shape, ())
        self.assertAllClose(loss_grad.shape, expected_grad_shape)

        self.assertFalse(gs.isclose(loss_value, 0.0))
        self.assertFalse(gs.isnan(loss_value))
        self.assertFalse(gs.all(gs.isclose(loss_grad, gs.zeros(expected_grad_shape))))
        self.assertTrue(gs.all(~gs.isnan(loss_grad)))

    @tests.conftest.autograd_and_torch_only
    def test_loss_minimization_extrinsic_euclidean(self):
        """Minimize loss from noiseless data."""
        pr = PolynomialRegression(self.eucl, order=self.order_eucl, regularization=0)

        def loss_of_param(param):
            return pr._loss(self.X_eucl, self.y_eucl, param, self.shape_eucl)

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param, to_numpy=True)
        initial_guess = gs.flatten(self.param_eucl_guess)

        # May need to enforce stricter tolerance for tol given larger parameter space
        res = minimize(
            objective_with_grad,
            initial_guess,
            method="CG",
            jac=True,
            tol=gs.atol / 100,
            options={"disp": True, "maxiter": 300},
        )

        self.assertAllClose(
            gs.array(res.x).shape[0], self.dim_eucl * (self.order_eucl + 1)
        )
        self.assertAllClose(res.fun, 0.0, atol=100 * gs.atol)

        # Cast required because minimization happens in scipy in float64
        param_hat = gs.cast(gs.array(res.x), self.param_eucl_true.dtype)

        intercept_hat, coef_hat = pr.split_parameters(param_hat)
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
            transported_coef_hat, self.coef_eucl_true, atol=1000 * gs.atol
        )

    @tests.conftest.autograd_and_torch_only
    def test_loss_minimization_extrinsic_hypersphere(self):
        """Minimize loss from noiseless data."""
        pr = PolynomialRegression(
            self.sphere, order=self.order_sphere, regularization=0
        )

        def loss_of_param(param):
            return pr._loss(self.X_sphere, self.y_sphere, param, self.shape_sphere)

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param, to_numpy=True)
        initial_guess = gs.flatten(self.param_sphere_guess)
        res = minimize(
            objective_with_grad,
            initial_guess,
            method="CG",
            jac=True,
            tol=gs.atol,
            options={"disp": True, "maxiter": 100},
        )
        self.assertAllClose(
            gs.array(res.x).shape, (self.shape_sphere[0] * (self.order_sphere + 1),)
        )
        self.assertAllClose(res.fun, 0.0, atol=5e-3)

        # Cast required because minimization happens in scipy in float64
        param_hat = gs.cast(gs.array(res.x), self.param_sphere_true.dtype)

        intercept_hat, coef_hat = pr.split_parameters(param_hat)
        intercept_hat = self.sphere.projection(intercept_hat)
        coef_hat = self.sphere.to_tangent(coef_hat, intercept_hat)
        self.assertAllClose(intercept_hat, self.intercept_sphere_true, atol=5e-2)

        # Coefficient must be parallel transported by log(intercept hat, intercept true)

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
        pr = PolynomialRegression(
            self.se2,
            metric=self.metric_se2,
            order=self.order_se2,
            center_X=False,
            method="extrinsic",
            max_iter=50,
            init_step_size=0.1,
            verbose=True,
        )

        def loss_of_param(param):
            return pr._loss(self.X_se2, self.y_se2, param, self.shape_se2)

        objective_with_grad = gs.autodiff.value_and_grad(loss_of_param, to_numpy=True)

        # Need longer optimization as more parameters in matrix space
        res = minimize(
            objective_with_grad,
            gs.flatten(self.param_se2_guess),
            method="CG",
            jac=True,
            tol=gs.atol,
            options={"disp": True, "maxiter": 100},
        )
        print(gs.array(res.x).shape)
        self.assertAllClose(
            gs.array(res.x).shape[0],
            (self.shape_se2[0] * self.shape_se2[1] * (self.order_se2 + 1)),
        )

        self.assertAllClose(res.fun, 0.0, atol=1e-6)

        # Cast required because minimization happens in scipy in float64
        param_hat = gs.cast(gs.array(res.x), self.param_se2_true.dtype)

        intercept_hat, coef_hat = pr.split_parameters(param_hat)
        intercept_hat = gs.reshape(intercept_hat, self.shape_se2)
        coef_hat = gs.reshape(coef_hat, (self.order_se2,) + self.shape_se2)

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
        pr = PolynomialRegression(
            self.eucl,
            metric=self.eucl.metric,
            order=self.order_eucl,
            center_X=False,
            method="extrinsic",
            max_iter=200,
            init_step_size=0.01,
            verbose=True,
            initialization="frechet",
            regularization=0.9,
        )

        print(self.param_eucl_true.shape)

        pr.fit(self.X_eucl, self.y_eucl, compute_training_score=True)

        training_score = pr.training_score_
        intercept_hat, coef_hat = pr.intercept_, pr.coef_
        self.assertAllClose(intercept_hat.shape, self.shape_eucl)
        self.assertAllClose(coef_hat.shape, (self.order_eucl,) + self.shape_eucl)
        self.assertAllClose(training_score, 1.0, atol=500 * gs.atol)
        self.assertAllClose(intercept_hat, self.intercept_eucl_true, atol=1e-5)

        tangent_vec_of_transport = self.eucl.metric.log(
            self.intercept_eucl_true, base_point=intercept_hat
        )

        transported_coef_hat = self.eucl.metric.parallel_transport(
            tangent_vec=coef_hat,
            base_point=intercept_hat,
            direction=tangent_vec_of_transport,
        )

        self.assertAllClose(transported_coef_hat, self.coef_eucl_true, atol=1e-3)

    @tests.conftest.autograd_and_torch_only
    def test_fit_extrinsic_hypersphere(self):
        pr = PolynomialRegression(
            self.sphere,
            metric=self.sphere.metric,
            order=self.order_sphere,
            center_X=False,
            method="extrinsic",
            max_iter=200,
            init_step_size=0.01,
            verbose=True,
            initialization="frechet",
            regularization=0.5,
        )

        pr.fit(self.X_sphere, self.y_sphere, compute_training_score=True)

        training_score = pr.training_score_
        intercept_hat, coef_hat = pr.intercept_, pr.coef_

        self.assertAllClose(intercept_hat.shape, self.shape_sphere)
        self.assertAllClose(coef_hat.shape, (self.order_sphere,) + self.shape_sphere)
        self.assertAllClose(training_score, 1.0, atol=1.5e-2)
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
        pr = PolynomialRegression(
            self.se2,
            metric=self.metric_se2,
            order=self.order_se2,
            center_X=False,
            method="extrinsic",
            max_iter=100,
            init_step_size=0.1,
            verbose=True,
            initialization="warm_start",
        )

        pr.fit(self.X_se2, self.y_se2, compute_training_score=True)
        intercept_hat, coef_hat = pr.intercept_, pr.coef_
        training_score = pr.training_score_

        self.assertAllClose(intercept_hat.shape, self.shape_se2)
        self.assertAllClose(coef_hat.shape, (self.order_se2,) + self.shape_se2)
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
        pr = PolynomialRegression(
            self.eucl,
            metric=self.eucl.metric,
            order=self.order_eucl,
            center_X=False,
            method="riemannian",
            initialization="frechet",
            max_iter=1000,
            tol=1e-7,
            init_step_size=0.01,
            verbose=True,
        )

        pr.fit(self.X_eucl, self.y_eucl, compute_training_score=True)
        intercept_hat, coef_hat = pr.intercept_, pr.coef_
        training_score = pr.training_score_

        self.assertAllClose(intercept_hat.shape, self.shape_eucl)
        self.assertAllClose(coef_hat.shape, (self.order_eucl,) + self.shape_eucl)

        self.assertAllClose(training_score, 1.0, atol=0.1)
        self.assertAllClose(intercept_hat, self.intercept_eucl_true, atol=1e-2)

        tangent_vec_of_transport = self.eucl.metric.log(
            self.intercept_eucl_true, base_point=intercept_hat
        )

        transported_coef_hat = self.eucl.metric.parallel_transport(
            tangent_vec=coef_hat,
            base_point=intercept_hat,
            direction=tangent_vec_of_transport,
        )

        self.assertAllClose(transported_coef_hat, self.coef_eucl_true, atol=0.5)

    @tests.conftest.autograd_and_torch_only
    def test_fit_riemannian_hypersphere(self):
        pr = PolynomialRegression(
            self.sphere,
            metric=self.sphere.metric,
            order=self.order_sphere,
            center_X=False,
            method="riemannian",
            max_iter=200,
            tol=1e-5,
            initialization="frechet",
            init_step_size=0.01,
            verbose=True,
        )

        pr.fit(self.X_sphere, self.y_sphere, compute_training_score=True)
        intercept_hat, coef_hat = pr.intercept_, pr.coef_
        training_score = pr.training_score_

        self.assertAllClose(intercept_hat.shape, self.shape_sphere)
        self.assertAllClose(coef_hat.shape, (self.order_sphere,) + self.shape_sphere)

        self.assertAllClose(training_score, 1.0, atol=0.1)
        self.assertAllClose(intercept_hat, self.intercept_sphere_true, atol=0.1)

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
        pr = PolynomialRegression(
            self.se2,
            metric=self.metric_se2,
            order=self.order_se2,
            center_X=False,
            method="riemannian",
            max_iter=500,
            tol=1e-6,
            init_step_size=0.01,
            verbose=True,
            initialization="frechet",
        )

        pr.fit(self.X_se2, self.y_se2, compute_training_score=True)
        intercept_hat, coef_hat = pr.intercept_, pr.coef_
        training_score = pr.training_score_

        self.assertAllClose(intercept_hat.shape, self.shape_se2)
        self.assertAllClose(coef_hat.shape, (self.order_se2,) + self.shape_se2)
        self.assertAllClose(training_score, 1.0, atol=1e-2)
        self.assertAllClose(intercept_hat, self.intercept_se2_true, atol=0.6)

        tangent_vec_of_transport = self.se2.metric.log(
            self.intercept_se2_true, base_point=intercept_hat
        )

        transported_coef_hat = self.se2.metric.parallel_transport(
            tangent_vec=coef_hat,
            base_point=intercept_hat,
            direction=tangent_vec_of_transport,
        )

        self.assertAllClose(transported_coef_hat, self.coef_se2_true, atol=0.6)
