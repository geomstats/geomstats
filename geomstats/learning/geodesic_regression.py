"""Geodesic Regression"""

import logging
import math
from scipy.optimize import minimize
from sklearn.base import BaseEstimator

import geomstats.backend as gs
import geomstats.errors
from geomstats.learning.frechet_mean import FrechetMean


class GeodesicRegression(BaseEstimator):
    def __init__(
            self, space, metric=None, center_data=True, algorithm='extrinsic',
            max_iter=100, verbose=False, learning_rate=.1, tol=1e-5):
        if metric is None:
            metric = space.metric
        self.metric = metric
        self.space = space
        self.intercept_ = None
        self.coef_ = None
        self.center_data = center_data
        self.mean_ = None
        self.training_score_ = None
        geomstats.errors.check_parameter_accepted_values(
            algorithm, 'algorithm', ['extrinsic', 'riemannian'])
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.tol = tol

    def _model(self, x, tangent_vec, base_point):
        times = x[:, None] if self.metric.default_point_type == 'vector' else\
            x[:, None, None]
        return self.metric.exp(times * tangent_vec[None], base_point)

    def _loss(self, x, y, parameter, shape, weights=None):
        p, v = gs.split(parameter, 2)
        p = gs.reshape(p, shape)
        v = gs.reshape(v, shape)
        base_point = self.space.projection(p)
        tangent_vec = self.space.to_tangent(v, base_point)
        distances = self.metric.squared_dist(
            self._model(x, tangent_vec, base_point), y)
        if weights is None:
            weights = 1.
        return 1. / 2. * gs.sum(weights * distances)

    def fit(self, X, y, weights=None, compute_training_score=False):
        if self.algorithm == 'extrinsic':
            return self._fit_extrinsic(X, y, weights, compute_training_score)
        if self.algorithm == 'riemannian':
            return self._fit_riemannian(X, y, weights, compute_training_score)

    def _fit_extrinsic(self, X, y, weights=None, compute_training_score=False):
        shape = (
            y.shape[-1:] if self.space.default_point_type == 'vector' else
            y.shape[-2:])

        times = gs.copy(X)
        if self.center_data:
            self.mean_ = gs.mean(X)
            times -= self.mean_

        initial_guess = gs.flatten(gs.stack([
            gs.random.normal(size=shape), gs.random.normal(size=shape)]))
        objective_with_grad = gs.autograd.value_and_grad(
            lambda param: self._loss(times, y, param, shape, weights))

        res = minimize(
            objective_with_grad, initial_guess, method='CG', jac=True,
            options={'disp': self.verbose, 'maxiter': self.max_iter},
            tol=self.tol)

        intercept_hat, beta_hat = gs.split(gs.array(res.x), 2)
        intercept_hat = gs.reshape(intercept_hat, shape)
        beta_hat = gs.reshape(beta_hat, shape)
        self.intercept_ = self.space.projection(intercept_hat)
        self.coef_ = self.space.to_tangent(beta_hat, intercept_hat)

        if compute_training_score:
            variance = gs.sum(self.metric.squared_dist(y, self.intercept_))
            self.training_score_ = 1 - 2 * res.fun / variance

        return self

    def _fit_riemannian(self, X, y, weights=None, compute_training_score=False):
        shape = (
            y.shape[-1:] if self.space.default_point_type == 'vector' else
            y.shape[-2:])

        times = gs.copy(X)
        if self.center_data:
            self.mean_ = gs.mean(X)
            times -= self.mean_

        if hasattr(self.metric, 'parallel_transport'):
            def vector_transport(tan_a, tan_b, base_point, _):
                return self.metric.parallel_transport(tan_a, tan_b, base_point)
        else:
            def vector_transport(tan_a, _, __, point):
                return self.space.to_tangent(tan_a, point)

        objective_with_grad = gs.autograd.value_and_grad(
            lambda params: self._loss(times, y, params, shape, weights))

        lr = self.learning_rate
        intercept_hat = intercept_hat_new = y[0]
        beta_hat = beta_hat_new = self.space.to_tangent(
            gs.random.normal(size=shape), intercept_hat)
        param = gs.vstack(
            [gs.flatten(intercept_hat), gs.flatten(beta_hat)])
        current_loss = math.inf
        current_iter = 0
        for i in range(self.max_iter):
            loss, grad = objective_with_grad(param)
            if loss > current_loss and i > 0:
                lr /= 2
            else:
                if not current_iter % 5:
                    lr *= 2
                beta_hat = beta_hat_new
                intercept_hat = intercept_hat_new
                current_iter += 1
            if abs(loss - current_loss) < self.tol:
                break
            print(self.space.belongs(intercept_hat))

            grad_p, grad_v = gs.split(grad, 2)
            riemannian_grad_p = self.space.to_tangent(
                gs.reshape(grad_p, shape), intercept_hat)
            riemannian_grad_v = self.space.to_tangent(
                gs.reshape(grad_v, shape), intercept_hat)
            print(i, - lr * gs.reshape(grad_p, shape), intercept_hat)

            intercept_hat_new = self.metric.exp(
                - lr * riemannian_grad_p, intercept_hat)
            beta_hat_new = vector_transport(
                beta_hat - lr * riemannian_grad_v,
                - lr * riemannian_grad_p, intercept_hat, intercept_hat_new)

            param = gs.vstack(
                [gs.flatten(intercept_hat_new), gs.flatten(beta_hat_new)])

            current_loss = loss

        self.intercept_ = self.space.projection(intercept_hat)
        self.coef_ = self.space.to_tangent(beta_hat, intercept_hat)

        if self.verbose:
            logging.info(f'Number of iteration: {current_iter},'
                         f' loss at termination: {current_loss}')
        if compute_training_score:
            variance = gs.sum(self.metric.squared_dist(y, self.intercept_))
            self.training_score_ = 1 - 2 * current_loss / variance

        return self

    def predict(self, X, y=None):
        times = gs.copy(X)

        if self.center_data:
            times -= self.mean_

        if self.coef_ is None:
            raise RuntimeError('Fit method must be called before transform')

        return self._model(times, self.coef_, self.intercept_)

    def score(self, X, y, weights=None):
        y_pred = self.predict(X)
        if weights is None:
            weights = 1.

        mean = FrechetMean(self.metric).fit(y).estimate_
        numerator = gs.sum(weights * self.metric.squared_dist(y, y_pred))
        denominator = gs.sum(weights * self.metric.squared_dist(
            y, mean))

        return 1 - numerator / denominator if denominator != 0 else 0.
