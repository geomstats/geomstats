r"""Polynomial Regression.

Lead author: Arka Mallela.

The generative model of the data is:
:math:`Z = Exp_{\beta_0}(\sum_{k=1}^{K}\beta_k.X^k)` and :math:`Y = Exp_Z(\epsilon)`
where:

- :math:`Exp` denotes the Riemannian exponential,
- :math:`\beta_0` is called the intercept,
  and is a point on the manifold,
- :math:`\beta_k` is called the coefficient, of :math:`X^k` power term
  and is a tangent vector to the manifold at :math:`\beta_0`,
- :math:`K` denotes the order of the polynomial,
- :math:`\epsilon \sim N(0, 1)` is a standard Gaussian noise,
- :math:`X` is the input, :math:`Y` is the target.

The polynomial regression method:

- estimates :math:`\beta_0`, and each :math:`\beta_k`,
- predicts :math:`\hat{y}` from input :math:`X`.
"""
import logging
import math

from scipy.optimize import minimize
from sklearn.base import BaseEstimator

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.matrices import Matrices
from geomstats.learning.frechet_mean import FrechetMean


class PolynomialRegression(BaseEstimator):
    r"""PolynomialRegression.

      The generative model of the data is:
      :math:`Z = Exp_{\beta_0}(\sum_{k=1}^{K}\beta_k.X^k)` and :math:`Y = Exp_Z(\epsilon)`
      where:

      - :math:`Exp` denotes the Riemannian exponential,
      - :math:`\beta_0` is called the intercept,
        and is a point on the manifold,
      - :math:`\beta_1` is called the coefficient,
        and is a tangent vector to the manifold at :math:`\beta_0`,
    - :math:`K` denotes the order of the polynomial,
      - :math:`\epsilon \sim N(0, 1)` is a standard Gaussian noise,
      - :math:`X` is the input, :math:`Y` is the target.

      The polynomial regression method:

      - estimates :math:`\beta_0`, and each :math:`\beta_k`,
      - predicts :math:`\hat{y}` from input :math:`X`.

      Parameters
      ----------
      space : Manifold
          Manifold.
      metric : RiemannianMetric
          Riemannian metric.
      center_X : bool
          Subtract mean to X as a preprocessing.
      order : int
          Order of polynomial to fit. Mandatory, maximum is 5 (for now)
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
        order,
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

        # For now, only allow for fitting through quintic polynomials
        geomstats.errors.check_parameter_accepted_values(
            order, "order", [i + 1 for i in range(0, 5)]
        )
        self.order = order

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
        """Compute the generative model of the polynomial regression.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[...,}]
            Training input samples.
        coef : array-like, shape=[..., {dim, [n,n]}]
            Coefficient of the polynomial regression.
        intercept : array-like, shape=[..., {dim, [n,n]}]
            Intercept of the polynomial regression.

        Returns
        -------
        _ : array-like, shape=[..., {dim, [n,n]}]
            Value on the manifold output by the generative model.
        """
        X_copy = X[:, None]

        # Pads additional dimensions as needed
        # generate power matrix - n_samples x order
        # This is a matrix where columns are powers of X
        X_powers = gs.hstack([X_copy**k for k in range(1, self.order + 1)])

        # Reshape twice to do mat mul between 2D arrays
        tangent_vec = gs.reshape(
            Matrices.mul(X_powers, gs.reshape(coef, (self.order, -1))),
            (-1,) + tuple(coef.shape)[1:],
        )
        return self.metric.exp(tangent_vec=tangent_vec, base_point=intercept)

    def _loss(self, X, y, param, shape, weights=None):
        """Compute the loss associated to the polynomial regression.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[...,}]
            Training input samples.
        y : array-like, shape=[..., {dim, [n,n]}]
            Training target values.
        param : array-like, shape=[order + 1, {dim, [n,n]}]
            Parameters intercept and coef of the polynomial regression,
            vertically stacked.
        weights : array-like, shape=[...,]
            Weights associated to the points.
            Optional, default: None.

        Returns
        -------
        _ : float
            Loss.
        """
        intercept, coef = self._split_parameters(param, shape)

        intercept = gs.cast(intercept, dtype=y.dtype)
        coef = gs.cast(coef, dtype=y.dtype)
        if self.method == "extrinsic":
            base_point = self.space.projection(intercept)
            penalty = self.regularization * gs.sum((base_point - intercept) ** 2)
        else:
            base_point = intercept
            penalty = 0
        tangent_vec = self.space.to_tangent(coef, base_point)
        mses = self.metric.squared_dist(self._model(X, tangent_vec, base_point), y)
        if weights is None:
            weights = 1.0

        return 1.0 / 2.0 * gs.sum(weights * mses) + penalty

    @staticmethod
    def _split_parameters(param, shape=None):
        """Split parameter matrix into intercept and coeff.

        Split parameters (order + 1 x dim) into intercept (-1 x shape)
        and coefficient matrix (-1 x shape).

        Parameters
        ----------
        param : array-like, shape=[order + 1, {dim, [n,n]}]
            Parameters intercept and coef of the polynomial regression,
            vertically stacked or in matrix form
        shape : array-like, shape=[order + 1, {order + 1}]
            Optional: default, None


        Returns
        -------
        intercept : array-like, shape=[{dim, [n,n]}]
             Value for the intercept.
        coef : array-like, shape=[{order, dim, [n,n]}]
            Initial value for the coefficient matrix.
        """
        if shape:
            return param[0].reshape(shape), param[1:].reshape((-1,) + shape)
        return param[0], param[1:]

    def _combine_parameters(self, intercept, coef):
        """Combine  intercept and coeff into param.

        Split parameters (order + 1 x dim) into intercept (1 x dim)
        and coefficient matrix (order x dim).

        Parameters
        ----------
        intercept : array-like, shape=[{dim, [n,n]}]
            Value for the intercept.
        coef : array-like, shape=[{order, dim, [n,n]}]
            Value for the coefficient matrix.

        Returns
        -------
        param : array-like, shape=[order + 1, {dim, [n,n]}]
            Parameters intercept and coef of the polynomial regression,
            vertically stacked or in matrix form

        """
        return gs.vstack([gs.flatten(intercept), gs.reshape(coef, (self.order, -1))])

    def fit(self, X, y, weights=None, compute_training_score=False):
        """Estimate the parameters of the polynomial regression.

        Estimate the intercept and the coefficient defining the
        polynomial regression model.

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
        polynomial regression model, using the extrinsic gradient.

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
        intercept_init = gs.reshape(intercept_init, (1,) + shape)
        # Need to reshape for matrix manifolds

        intercept_hat = self.space.projection(intercept_init)
        coef_init = gs.reshape(coef_init, (self.order,) + shape)
        coef_hat = self.space.to_tangent(coef_init, intercept_hat)
        initial_guess = self._combine_parameters(intercept_hat, coef_hat)

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

        intercept_hat, coef_hat = self._split_parameters(gs.array(res.x))
        intercept_hat = gs.reshape(intercept_hat, shape)
        intercept_hat = gs.cast(intercept_hat, dtype=y.dtype)
        coef_hat = gs.reshape(coef_hat, (self.order,) + shape)
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
        coef : array-like, shape=[{order, dim, [n,n]}]
            Initial value for the coefficient matrix.
        """
        init = self.initialization
        shape = (
            y.shape[-1:] if self.space.default_point_type == "vector" else y.shape[-2:]
        )
        if isinstance(init, str):
            if init == "random":
                return self._split_parameters(
                    gs.random.normal(size=(1 + self.order,) + shape)
                )
            if init == "frechet":
                mean = FrechetMean(self.metric, verbose=self.verbose).fit(y).estimate_
                return mean, gs.zeros((self.order,) + shape)
            if init == "data":
                return gs.random.choice(y, 1)[0], gs.random.normal(
                    size=(self.order,) + shape
                )
            if init == "warm_start":
                if self.intercept_ is not None:
                    return self.intercept_, self.coef_
                return self._split_parameters(
                    gs.random.normal(size=(1 + self.order,) + shape)
                )
            raise ValueError(
                "The initialization string must be one of "
                "random, frechet, data or warm_start"
            )
        return init

    def _fit_riemannian(self, X, y, weights=None, compute_training_score=False):
        """Estimate the parameters using a Riemannian gradient descent.

        Estimate the intercept and the coefficient defining the
        polynomial regression model, using the Riemannian gradient.

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
        param = self._combine_parameters(intercept_hat, coef_hat)
        current_loss = [math.inf]
        current_grad = gs.zeros_like(param)
        current_iter = i = 0

        for i in range(self.max_iter):
            loss, grad = objective_with_grad(param)
            if gs.any(gs.isnan(grad)):
                logging.warning("NaN encountered in gradient at iter %s", current_iter)
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
                    logging.info("Tolerance threshold reached at iter %s", current_iter)
                break

            grad_intercept, grad_coef = self._split_parameters(grad)

            riem_grad_intercept = self.space.to_tangent(
                gs.reshape(grad_intercept, shape), intercept_hat
            )
            riem_grad_coef = self.space.to_tangent(
                gs.reshape(grad_coef, (self.order,) + shape), intercept_hat
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

            param = self._combine_parameters(intercept_hat_new, coef_hat_new)

            current_loss.append(loss)
            current_grad = grad

        self.intercept_ = self.space.projection(intercept_hat)
        self.coef_ = self.space.to_tangent(coef_hat, self.intercept_)

        if self.verbose:
            logging.info(
                "Number of gradient evaluations: %s, "
                "Number of gradient iterations: %s"
                " loss at termination: %s",
                i,
                current_iter,
                current_loss[-1],
            )
        if compute_training_score:
            variance = gs.sum(self.metric.squared_dist(y, self.intercept_))
            self.training_score_ = 1 - 2 * current_loss[-1] / variance

        return self

    def predict(self, X):
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
        if self.coef_ is None:
            raise RuntimeError("Fit method must be called before predict.")

        if self.center_X:
            X -= self.mean_

        return self._model(X, self.coef_, self.intercept_)

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
