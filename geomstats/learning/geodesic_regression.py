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

from scipy.optimize import OptimizeResult
from sklearn.base import BaseEstimator

import geomstats.backend as gs
import geomstats.errors as error
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.numerics.optimizers import ScipyMinimize


class RiemannianGradientDescent:
    """Riemannian gradient descent."""

    def __init__(self, max_iter=100, init_step_size=0.1, tol=1e-5, verbose=False):
        self.max_iter = max_iter
        self.init_step_size = init_step_size
        self.verbose = verbose
        self.tol = tol
        self.jac = "autodiff"

    def _handle_jac(self, fun):
        if self.jac == "autodiff":

            def fun_(x):
                value, grad = gs.autodiff.value_and_grad(fun, to_numpy=True)(
                    gs.from_numpy(x)
                )
                return value, grad

        else:
            raise NotImplementedError("For now only working with autodiff.")

        return fun_

    def _get_vector_transport(self, space):
        if hasattr(space.metric, "parallel_transport"):

            def vector_transport(tan_a, tan_b, base_point, _):
                return space.metric.parallel_transport(tan_a, base_point, tan_b)

        else:

            def vector_transport(tan_a, _, __, point):
                return space.to_tangent(tan_a, point)

        return vector_transport

    def minimize(self, space, fun, x0):
        """Perform gradient descent."""
        fun = self._handle_jac(fun)
        vector_transport = self._get_vector_transport(space)

        lr = self.init_step_size

        intercept_init, coef_init = gs.split(x0, 2)
        intercept_init = gs.reshape(intercept_init, space.shape)
        coef_init = gs.reshape(coef_init, space.shape)

        intercept_hat = intercept_hat_new = space.projection(intercept_init)
        coef_hat = coef_hat_new = space.to_tangent(coef_init, intercept_hat)
        param = gs.vstack([gs.flatten(intercept_hat), gs.flatten(coef_hat)])

        current_loss = math.inf
        current_grad = gs.zeros_like(param)
        current_iter = i = 0

        for i in range(self.max_iter):
            loss, grad = fun(param)
            if gs.any(gs.isnan(grad)):
                logging.warning(f"NaN encountered in gradient at iter {current_iter}")
                lr /= 2
                grad = current_grad
            elif loss >= current_loss and i > 0:
                lr /= 2
            else:
                if not current_iter % 5:
                    lr *= 2
                coef_hat = coef_hat_new
                intercept_hat = intercept_hat_new
                current_iter += 1
            if abs(loss - current_loss) < self.tol:
                if self.verbose:
                    logging.info(f"Tolerance threshold reached at iter {current_iter}")
                break

            grad_intercept, grad_coef = gs.split(grad, 2)
            riem_grad_intercept = space.to_tangent(
                gs.reshape(grad_intercept, space.shape), intercept_hat
            )
            riem_grad_coef = space.to_tangent(
                gs.reshape(grad_coef, space.shape), intercept_hat
            )

            intercept_hat_new = space.metric.exp(
                -lr * riem_grad_intercept, intercept_hat
            )
            coef_hat_new = vector_transport(
                coef_hat - lr * riem_grad_coef,
                -lr * riem_grad_intercept,
                intercept_hat,
                intercept_hat_new,
            )

            param = gs.vstack([gs.flatten(intercept_hat_new), gs.flatten(coef_hat_new)])

            current_loss = loss
            current_grad = grad

        if self.verbose:
            logging.info(
                f"Number of gradient evaluations: {i}, "
                f"Number of gradient iterations: {current_iter}"
                f" loss at termination: {current_loss}"
            )

        return OptimizeResult(fun=loss, x=param, nit=current_iter)


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
        Equipped manifold.
    center_X : bool
        Subtract mean to X as a preprocessing.
    method : str, {\'extrinsic\', \'riemannian\'}
        Gradient descent method.
        Optional, default: extrinsic.
    initialization : str or array-like,
        {'random', 'data', 'frechet', warm_start'}
        Initial values of the parameters for the optimization,
        or initialization method.
        Optional, default: 'random'
    regularization : float
        Weight on the constraint for the intercept to lie on the manifold in
        the extrinsic optimization scheme. An L^2 constraint is applied.
        Optional, default: 1.
    compute_training_score : bool
        Whether to compute R^2.
        Optional, default: False.

    Notes
    -----
    * Required metric methods:
        * all: `exp`, `squared_dist`
        * if `riemannian`: `parallel transport` or `to_tangent`
    """

    def __init__(
        self,
        space,
        center_X=True,
        method="extrinsic",
        initialization="random",
        regularization=1.0,
        compute_training_score=False,
    ):
        self.space = space
        self.center_X = center_X
        self._method = None
        self.method = method
        self.initialization = initialization
        self.regularization = regularization
        self.compute_training_score = compute_training_score

        self.intercept_ = None
        self.coef_ = None
        self.mean_ = None
        self.training_score_ = None

        self.mean_estimator = FrechetMean(self.space)

    def set(self, **kwargs):
        """Set optimizer parameters.

        Especially useful for one line instantiations.
        """
        for param_name, value in kwargs.items():
            if not hasattr(self.optimizer, param_name):
                raise ValueError(f"Unknown parameter {param_name}.")

            setattr(self.optimizer, param_name, value)
        return self

    @property
    def method(self):
        """Gradient descent method."""
        return self._method

    @method.setter
    def method(self, value):
        """Gradient descent method."""
        error.check_parameter_accepted_values(
            value, "method", ["extrinsic", "riemannian"]
        )
        if value == self._method:
            return

        self._method = value

        tol = 1e-5
        max_iter = 100
        if value == "extrinsic":
            optimizer = ScipyMinimize(
                method="CG",
                options={"disp": False, "maxiter": max_iter},
                tol=tol,
            )

        else:
            optimizer = RiemannianGradientDescent(
                max_iter=max_iter,
                init_step_size=0.1,
                tol=tol,
                verbose=False,
            )

        self.optimizer = optimizer

    def _model(self, X, coef, intercept):
        """Compute the generative model of the geodesic regression.

        Parameters
        ----------
        X : array-like, shape=[n_samples,]
            Training input samples.
        coef : array-like, shape=[{dim, [n,n]}]
            Coefficient of the geodesic regression.
        intercept : array-like, shape=[{dim, [n,n]}]
            Intercept of the geodesic regression.

        Returns
        -------
        _ : array-like, shape=[..., {dim, [n,n]}]
            Value on the manifold output by the generative model.
        """
        return self.space.metric.exp(gs.einsum("n,...->n...", X, coef), intercept)

    def _loss(self, X, y, param, weights=None):
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
        intercept = gs.reshape(intercept, self.space.shape)
        coef = gs.reshape(coef, self.space.shape)

        if self.method == "extrinsic":
            base_point = self.space.projection(intercept)
            penalty = self.regularization * gs.sum((base_point - intercept) ** 2)
        else:
            base_point = intercept
            penalty = 0.0

        tangent_vec = self.space.to_tangent(coef, base_point)
        distances = self.space.metric.squared_dist(
            self._model(X, tangent_vec, base_point), y
        )
        if weights is None:
            weights = 1.0
        return 1.0 / 2.0 * gs.sum(weights * distances) + penalty

    def _initialize_parameters(self, y):
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
        shape = self.space.shape

        if isinstance(init, str):
            if init == "random":
                return gs.random.normal(size=(2,) + shape)
            if init == "frechet":
                mean = self.mean_estimator.fit(y).estimate_
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

    def fit(self, X, y, weights=None):
        """Estimate the parameters of the geodesic regression.

        Estimate the intercept and the coefficient defining the
        geodesic regression model.

        Parameters
        ----------
        X : array-like, shape=[n_samples,]
            Training input samples.
        y : array-like, shape[n_samples, {dim, [n,n]}]
            Training target values.
        weights : array-like, shape=[n_samples]
            Weights associated to the points.
            Optional, default: None.

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
            res = self._fit_extrinsic(times, y, weights)
        if self.method == "riemannian":
            res = self._fit_riemannian(times, y, weights)

        intercept_hat, coef_hat = gs.split(res.x, 2)
        intercept_hat = gs.reshape(intercept_hat, self.space.shape)
        coef_hat = gs.reshape(coef_hat, self.space.shape)

        self.intercept_ = self.space.projection(intercept_hat)
        self.coef_ = self.space.to_tangent(coef_hat, self.intercept_)

        if self.compute_training_score:
            variance = gs.sum(self.space.metric.squared_dist(y, self.intercept_))
            self.training_score_ = 1 - 2 * res.fun / variance

        return self

    def _fit_extrinsic(self, X, y, weights=None):
        """Estimate the parameters using the extrinsic gradient descent.

        Estimate the intercept and the coefficient defining the
        geodesic regression model, using the extrinsic gradient.

        Parameters
        ----------
        X : array-like, shape=[n_samples,]
            Training input samples.
        y : array-like, shape=[n_samples, {dim, [n,n]}]
            Training target values.
        weights : array-like, shape=[n_samples,]
            Weights associated to the points.
            Optional, default: None.

        Returns
        -------
        res : OptimizeResult
            Scipy's optimize result.
        """
        intercept_init, coef_init = self._initialize_parameters(y)
        intercept_hat = self.space.projection(intercept_init)
        coef_hat = self.space.to_tangent(coef_init, intercept_hat)
        initial_guess = gs.hstack([gs.flatten(intercept_hat), gs.flatten(coef_hat)])

        objective_with_grad = lambda param: self._loss(X, y, param, weights)

        return self.optimizer.minimize(
            objective_with_grad,
            initial_guess,
        )

    def _fit_riemannian(self, X, y, weights=None):
        """Estimate the parameters using a Riemannian gradient descent.

        Estimate the intercept and the coefficient defining the
        geodesic regression model, using the Riemannian gradient.

        Parameters
        ----------
        X : array-like, shape=[n_samples,]
            Training input samples.
        y : array-like, shape=[n_samples, {dim, [n,n]}]
            Training target values.
        weights : array-like, shape=[n_samples,]
            Weights associated to the points.
            Optional, default: None.

        Returns
        -------
        res : OptimizeResult
            Scipy's optimize result.
        """
        objective_with_grad = lambda params: self._loss(X, y, params, weights)

        intercept_init, coef_init = self._initialize_parameters(y)
        x0 = gs.vstack([gs.flatten(intercept_init), gs.flatten(coef_init)])

        return self.optimizer.minimize(self.space, objective_with_grad, x0)

    def predict(self, X):
        """Predict the manifold value for each input.

        Parameters
        ----------
        X : array-like, shape=[n_samples,]
            Input data.

        Returns
        -------
        y : array-like, shape=[n_samples, {dim, [n,n]}]
            Training target values.
        """
        if self.coef_ is None:
            raise RuntimeError("Fit method must be called before predict.")

        times = gs.copy(X)

        if self.center_X:
            times = times - self.mean_

        return self._model(times, self.coef_, self.intercept_)

    def score(self, X, y, weights=None):
        """Compute training score.

        Compute the training score defined as R^2.

        Parameters
        ----------
        X : array-like, shape=[n_samples,]
            Training input samples.
        y : array-like, shape=[n_samples, {dim, [n,n]}]
            Training target values.
        weights : array-like, shape=[n_samples,]
            Weights associated to the points.
            Optional, default: None.

        Returns
        -------
        score : float
            Training score.
        """
        y_pred = self.predict(X)
        if weights is None:
            weights = 1.0

        mean = self.mean_estimator.fit(y).estimate_
        numerator = gs.sum(weights * self.space.metric.squared_dist(y, y_pred))
        denominator = gs.sum(weights * self.space.metric.squared_dist(y, mean))

        return 1 - numerator / denominator if denominator != 0 else 0.0
