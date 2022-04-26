r"""Geodesic Regression.

Lead author: Nicolas Guigui.

The generative model of the data is:
    :math:`Z = Exp_{\beta_0}(\beta_1.X)` and :math:`Y = Exp_Z(\epsilon)`
    where:
    - :math:`Exp` denotes the Riemannian exponential,
    - :math:`\beta_0` is called the intercept,
    and is a point on the manifold,
    - :math:`\beta_1` is called the coefficient,
    and is a tangent vector to the manifold at :math:`\beta_0`,
    - :math:`\epsilon \sim N(0, 1)` is a standard Gaussian noise,
    - :math:`X` is the input, :math:`Y` is the target.

The geodesic regression method:
- estimates :math:`\beta_0, \beta_1`,
- predicts :math:`\hat{y}` from input :math:`X`.
"""

import logging
import math

from scipy.optimize import minimize
from sklearn.base import BaseEstimator

import geomstats.backend as gs
import geomstats.errors
from geomstats.learning.frechet_mean import FrechetMean


class GeodesicRegression(BaseEstimator):
    r"""Geodesic Regression.

    The generative model of the data is:
        :math:`Z = Exp_{\beta_0}(\beta_1.X)` and :math:`Y = Exp_Z(\epsilon)`
        where:
        - :math:`Exp` denotes the Riemannian exponential,
        - :math:`\beta_0` is called the intercept,
        and is a point on the manifold,
        - :math:`\beta_1` is called the coefficient,
        and is a tangent vector to the manifold at :math:`\beta_0`,
        - :math:`\epsilon \sim N(0, 1)` is a standard Gaussian noise,
        - :math:`X` is the input, :math:`Y` is the target.

    The geodesic regression method:
    - estimates :math:`\beta_0, \beta_1`,
    - predicts :math:`\hat{y}` from input :math:`X`.

    Parameters
    ----------
    space : Manifold
        Manifold.
    metric : RiemannianMetric
        Riemannian metric.
    center_X : bool
        Subtract mean to X as a preprocessing.
    method : str, {\'extrinsic\', \'riemannian\'}
        Gradient descent method.
        Optional, default: extrinsic.
    max_iter : int
        Maximum number of iterations for gradient descent.
        Optional, default: 100.
    init_step_size : float
        Initial learning rate for gradient descent.
        Optional, default: 0.1
    tol : float
        Tolerance for loss minimization.
        Optional, default: 1e-5
    verbose : bool
        Verbose option.
        Optional, default: False.
    initialization : str or array-like,
        {'random', 'data', 'frechet', warm_start'}
        Initial values of the parameters for the optimization,
        or initialization method.
        Optional, default: 'random'
    regularization : float
        Weight on the constraint for the intercept to lie on the manifold in
        the extrinsic optimization scheme. An L^2 constraint is applied.
        Optional, default: 1.
    """

    def __init__(
        self,
        space,
        metric=None,
        center_X=True,
        method="extrinsic",
        max_iter=100,
        init_step_size=0.1,
        tol=1e-5,
        verbose=False,
        initialization="random",
        regularization=1.0,
    ):
        if metric is None:
            metric = space.metric
        self.metric = metric
        self.space = space
        self.intercept_ = None
        self.coef_ = None
        self.center_X = center_X
        self.mean_ = None
        self.training_score_ = None
        geomstats.errors.check_parameter_accepted_values(
            method, "method", ["extrinsic", "riemannian"]
        )
        self.method = method
        self.max_iter = max_iter
        self.verbose = verbose
        self.init_step_size = init_step_size
        self.tol = tol
        self.initialization = initialization
        self.regularization = regularization

    def _model(self, X, coef, intercept):
        """Compute the generative model of the geodesic regression.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[...,}]
            Training input samples.
        coef : array-like, shape=[..., {dim, [n,n]}]
            Coefficient of the geodesic regression.
        intercept : array-like, shape=[..., {dim, [n,n]}]
            Intercept of the geodesic regression.

        Returns
        -------
        _ : array-like, shape=[..., {dim, [n,n]}]
            Value on the manifold output by the generative model.
        """
        X_copy = (
            X[:, None]
            if self.metric.default_point_type == "vector"
            else X[:, None, None]
        )
        return self.metric.exp(X_copy * coef[None], intercept)

    def _loss(self, X, y, param, shape, weights=None):
        """Compute the loss associated to the geodesic regression.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[...,}]
            Training input samples.
        y : array-like, shape=[..., {dim, [n,n]}]
            Training target values.
        param : array-like, shape=[2, {dim, [n,n]}]
            Parameters intercept and coef of the geodesic regression,
            vertically stacked.
        weights : array-like, shape=[...,]
            Weights associated to the points.
            Optional, default: None.

        Returns
        -------
        _ : float
            Loss.
        """
        intercept, coef = gs.split(param, 2)
        intercept = gs.reshape(intercept, shape)
        coef = gs.reshape(coef, shape)
        intercept = gs.cast(intercept, dtype=y.dtype)
        coef = gs.cast(coef, dtype=y.dtype)
        if self.method == "extrinsic":
            base_point = self.space.projection(intercept)
            penalty = self.regularization * gs.sum((base_point - intercept) ** 2)
        else:
            base_point = intercept
            penalty = 0
        tangent_vec = self.space.to_tangent(coef, base_point)
        distances = self.metric.squared_dist(self._model(X, tangent_vec, base_point), y)
        if weights is None:
            weights = 1.0
        return 1.0 / 2.0 * gs.sum(weights * distances) + penalty

    def fit(self, X, y, weights=None, compute_training_score=False):
        """Estimate the parameters of the geodesic regression.

        Estimate the intercept and the coefficient defining the
        geodesic regression model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[...,}]
            Training input samples.
        y : array-like, shape=[..., {dim, [n,n]}]
            Training target values.
        weights : array-like, shape=[...,]
            Weights associated to the points.
            Optional, default: None.
        compute_training_score : bool
            Whether to compute R^2.
            Optional, default: False.

        Returns
        -------
        self : object
            Returns self.
        """
        times = gs.copy(X)
        if self.center_X:
            self.mean_ = gs.mean(X)
            times -= self.mean_

        if self.method == "extrinsic":
            return self._fit_extrinsic(times, y, weights, compute_training_score)
        if self.method == "riemannian":
            return self._fit_riemannian(times, y, weights, compute_training_score)

    def _fit_extrinsic(self, X, y, weights=None, compute_training_score=False):
        """Estimate the parameters using the extrinsic gradient descent.

        Estimate the intercept and the coefficient defining the
        geodesic regression model, using the extrinsic gradient.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[...,}]
            Training input samples.
        y : array-like, shape=[..., {dim, [n,n]}]
            Training target values.
        weights : array-like, shape=[...,]
            Weights associated to the points.
            Optional, default: None.
        compute_training_score : bool
            Whether to compute R^2.
            Optional, default: False.

        Returns
        -------
        self : object
            Returns self.
        """
        shape = (
            y.shape[-1:] if self.space.default_point_type == "vector" else y.shape[-2:]
        )

        intercept_init, coef_init = self.initialize_parameters(y)
        intercept_hat = self.space.projection(intercept_init)
        coef_hat = self.space.to_tangent(coef_init, intercept_hat)
        initial_guess = gs.vstack([gs.flatten(intercept_hat), gs.flatten(coef_hat)])

        objective_with_grad = gs.autodiff.value_and_grad(
            lambda param: self._loss(X, y, param, shape, weights), to_numpy=True
        )

        res = minimize(
            objective_with_grad,
            initial_guess,
            method="CG",
            jac=True,
            options={"disp": self.verbose, "maxiter": self.max_iter},
            tol=self.tol,
        )

        intercept_hat, coef_hat = gs.split(gs.array(res.x), 2)
        intercept_hat = gs.reshape(intercept_hat, shape)
        intercept_hat = gs.cast(intercept_hat, dtype=y.dtype)
        coef_hat = gs.reshape(coef_hat, shape)
        coef_hat = gs.cast(coef_hat, dtype=y.dtype)

        self.intercept_ = self.space.projection(intercept_hat)
        self.coef_ = self.space.to_tangent(coef_hat, self.intercept_)

        if compute_training_score:
            variance = gs.sum(self.metric.squared_dist(y, self.intercept_))
            self.training_score_ = 1 - 2 * res.fun / variance

        return self

    def initialize_parameters(self, y):
        """Set initial values for the parameters of the model.

        Set initial parameters for the optimization, depending on the value
        of the attribute `initialization`. The options are:
            - `random` : pick random numbers from a normal distribution,
            then project them to the manifold and the tangent space.
            - `frechet` : compute the Frechet mean of the target points
            - `data` : pick a random sample from the target points and a
            tangent vector with random coefficients.
            - `warm_start`: pick previous values of the parameters if the
            model was fitted before, otherwise behaves as `random`.

        Parameters
        ----------
        y: array-like, shape=[n_samples, {dim, [n,n]}]
            The target data, used for the option `data` and 'frechet'.

        Returns
        -------
        intercept : array-like, shape=[{dim, [n,n]}]
            Initial value for the intercept.
        coef : array-like, shape=[{dim, [n,n]}]
            Initial value for the coefficient.
        """
        init = self.initialization
        shape = (
            y.shape[-1:] if self.space.default_point_type == "vector" else y.shape[-2:]
        )
        if isinstance(init, str):
            if init == "random":
                return gs.random.normal(size=(2,) + shape)
            if init == "frechet":
                mean = FrechetMean(self.metric, verbose=self.verbose).fit(y).estimate_
                return mean, gs.zeros(shape)
            if init == "data":
                return gs.random.choice(y, 1)[0], gs.random.normal(size=shape)
            if init == "warm_start":
                if self.intercept_ is not None:
                    return self.intercept_, self.coef_
                return gs.random.normal(size=(2,) + shape)
            raise ValueError(
                "The initialization string must be one of "
                "random, frechet, data or warm_start"
            )
        return init

    def _fit_riemannian(self, X, y, weights=None, compute_training_score=False):
        """Estimate the parameters using a Riemannian gradient descent.

        Estimate the intercept and the coefficient defining the
        geodesic regression model, using the Riemannian gradient.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[...,}]
            Training input samples.
        y : array-like, shape=[..., {dim, [n,n]}]
            Training target values.
        weights : array-like, shape=[...,]
            Weights associated to the points.
            Optional, default: None.
        compute_training_score : bool
            Whether to compute R^2.
            Optional, default: False.

        Returns
        -------
        self : object
            Returns self.
        """
        shape = (
            y.shape[-1:] if self.space.default_point_type == "vector" else y.shape[-2:]
        )
        if hasattr(self.metric, "parallel_transport"):

            def vector_transport(tan_a, tan_b, base_point, _):
                return self.metric.parallel_transport(tan_a, base_point, tan_b)

        else:

            def vector_transport(tan_a, _, __, point):
                return self.space.to_tangent(tan_a, point)

        objective_with_grad = gs.autodiff.value_and_grad(
            lambda params: self._loss(X, y, params, shape, weights)
        )

        lr = self.init_step_size
        intercept_init, coef_init = self.initialize_parameters(y)
        intercept_hat = intercept_hat_new = self.space.projection(intercept_init)
        coef_hat = coef_hat_new = self.space.to_tangent(coef_init, intercept_hat)
        param = gs.vstack([gs.flatten(intercept_hat), gs.flatten(coef_hat)])
        current_loss = [math.inf]
        current_grad = gs.zeros_like(param)
        current_iter = i = 0
        for i in range(self.max_iter):
            loss, grad = objective_with_grad(param)
            if gs.any(gs.isnan(grad)):
                logging.warning(f"NaN encountered in gradient at iter {current_iter}")
                lr /= 2
                grad = current_grad
            elif loss >= current_loss[-1] and i > 0:
                lr /= 2
            else:
                if not current_iter % 5:
                    lr *= 2
                coef_hat = coef_hat_new
                intercept_hat = intercept_hat_new
                current_iter += 1
            if abs(loss - current_loss[-1]) < self.tol:
                if self.verbose:
                    logging.info(f"Tolerance threshold reached at iter {current_iter}")
                break

            grad_intercept, grad_coef = gs.split(grad, 2)
            riem_grad_intercept = self.space.to_tangent(
                gs.reshape(grad_intercept, shape), intercept_hat
            )
            riem_grad_coef = self.space.to_tangent(
                gs.reshape(grad_coef, shape), intercept_hat
            )

            intercept_hat_new = self.metric.exp(
                -lr * riem_grad_intercept, intercept_hat
            )
            coef_hat_new = vector_transport(
                coef_hat - lr * riem_grad_coef,
                -lr * riem_grad_intercept,
                intercept_hat,
                intercept_hat_new,
            )

            param = gs.vstack([gs.flatten(intercept_hat_new), gs.flatten(coef_hat_new)])

            current_loss.append(loss)
            current_grad = grad

        self.intercept_ = self.space.projection(intercept_hat)
        self.coef_ = self.space.to_tangent(coef_hat, self.intercept_)

        if self.verbose:
            logging.info(
                f"Number of gradient evaluations: {i}, "
                f"Number of gradient iterations: {current_iter}"
                f" loss at termination: {current_loss[-1]}"
            )
        if compute_training_score:
            variance = gs.sum(self.metric.squared_dist(y, self.intercept_))
            self.training_score_ = 1 - 2 * current_loss[-1] / variance

        return self

    def predict(self, X, y=None):
        """Predict the manifold value for each input.

        Parameters
        ----------
        X : array-like, shape=[...,
            Input data.

        Returns
        -------
        self : array-like, shape=[...,]
            Array of predicted cluster indices for each sample.
        """
        times = gs.copy(X)

        if self.center_X:
            times = times - self.mean_

        if self.coef_ is None:
            raise RuntimeError("Fit method must be called before predict.")

        return self._model(times, self.coef_, self.intercept_)

    def score(self, X, y, weights=None):
        """Compute training score.

        Compute the training score defined as R^2.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[...,}]
            Training input samples.
        y : array-like, shape=[..., {dim, [n,n]}]
            Training target values.
        weights : array-like, shape=[...,]
            Weights associated to the points.
            Optional, default: None.

        Returns
        -------
        _ : float
            Training score.
        """
        y_pred = self.predict(X)
        if weights is None:
            weights = 1.0

        mean = FrechetMean(self.metric, verbose=self.verbose).fit(y).estimate_
        numerator = gs.sum(weights * self.metric.squared_dist(y, y_pred))
        denominator = gs.sum(weights * self.metric.squared_dist(y, mean))

        return 1 - numerator / denominator if denominator != 0 else 0.0
