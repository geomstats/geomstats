"""Geodesic Regression"""

from scipy.optimize import minimize
from sklearn.base import BaseEstimator

import geomstats.backend as gs
from geomstats.learning.frechet_mean import FrechetMean


class GeodesicRegression(BaseEstimator):
    def __init__(self, space, metric=None, center_data=True):
        if metric is None:
            metric = space.metric
        self.metric = metric
        self.space = space
        self.intercept_ = None
        self.coef_ = None
        self.center_data = center_data
        self.mean_ = None
        self.training_score_ = None

    def _model(self, x, tangent_vec, base_point):
        return self.metric.exp(x[:, None] * tangent_vec, base_point)

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
            options={'disp': True, 'maxiter': 100})

        intercept_hat, beta_hat = gs.split(gs.array(res.x), 2)
        intercept_hat = gs.reshape(intercept_hat, shape)
        beta_hat = gs.reshape(beta_hat, shape)
        self.intercept_ = self.space.projection(intercept_hat)
        self.coef_ = self.space.to_tangent(beta_hat, intercept_hat)

        if compute_training_score:
            variance = gs.sum(self.metric.squared_dist(y, self.intercept_))
            self.training_score_ = 1 - 2 * res.fun / variance

        return self

    def predict(self, X, y=None):
        times = gs.copy(X)

        if self.center_data:
            times -= self.mean_

        if self.coef_ is None:
            raise RuntimeError('Fit method must be called before transform')

        return self.metric.exp(times[..., None] * self.coef_, self.intercept_)

    def score(self, X, y, weights=None):
        y_pred = self.predict(X)
        if weights is None:
            weights = 1.

        mean = FrechetMean(self.metric).fit(y).estimate_
        numerator = gs.sum(weights * self.metric.squared_dist(y, y_pred))
        denominator = gs.sum(weights * self.metric.squared_dist(
            y, mean))

        return 1 - numerator / denominator if denominator != 0 else 0.
