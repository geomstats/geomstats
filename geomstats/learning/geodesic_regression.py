"""Geodesic Regression."""

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

    def _model(self, X, tangent_vec, base_point):
        X = X[:, None] if self.metric.default_point_type == 'vector' else\
            X[:, None, None]
        return self.metric.exp(X * tangent_vec[None], base_point)

    def _loss(self, X, y, param, shape, weights=None):
        intercept, coef = gs.split(param, 2)
        intercept = gs.reshape(intercept, shape)
        coef = gs.reshape(coef, shape)
        intercept = gs.cast(intercept, dtype=y.dtype)
        coef = gs.cast(coef, dtype=y.dtype)
        base_point = self.space.projection(intercept)
        tangent_vec = self.space.to_tangent(coef, base_point)
        distances = self.metric.squared_dist(
            self._model(X, tangent_vec, base_point), y)
        if weights is None:
            weights = 1.
        return 1. / 2. * gs.sum(weights * distances)

    def fit(self, X, y, weights=None, compute_training_score=False):
        times = gs.copy(X)
        if self.center_data:
            self.mean_ = gs.mean(X)
            times -= self.mean_

        if self.algorithm == 'extrinsic':
            return self._fit_extrinsic(times, y, weights, compute_training_score)
        if self.algorithm == 'riemannian':
            return self._fit_riemannian(times, y, weights, compute_training_score)

    def _fit_extrinsic(self, X, y, weights=None, compute_training_score=False):
        shape = (
            y.shape[-1:] if self.space.default_point_type == 'vector' else
            y.shape[-2:])
        
        # vector = gs.array([
        #     [ 0.50198963,  0.59494365,  0.5332258 ],
        #     [-2.18947023,  1.1374106 ,  0.5482675 ],
        #     [ 0.91499351,  0.78945069,  0.87057936]])
        # vector2 = gs.array([
        #     [ 0.88005482,  1.20483007, -0.87634215],
        #     [-1.13142537, -1.51064679, -0.05944102],
        #     [ 1.32931231, -0.60150817,  0.65693497]])
        initial_guess = gs.flatten(gs.stack([
            gs.random.normal(size=shape), 
            gs.random.normal(size=shape)
        ]))
        # initial_guess = gs.flatten(gs.stack([
        #     vector, vector2]))
        objective_with_grad = gs.autodiff.value_and_grad(
            lambda param: self._loss(X, y, param, shape, weights),
            to_numpy=True)

        print("\n\nJust before minimize")
        print("time/data:")
        print(type(X))
        print(X.shape)
        print(X)
        print("target")
        print(type(y))
        print(y.shape)
        print(y)
        print(self.space.belongs(y))

        print("\n\n Objective with grad")
        val, grad = objective_with_grad(initial_guess)
        print(val)
        print(type(grad))
        print(grad.shape)
        print(grad)

        res = minimize(
            objective_with_grad, initial_guess, method='CG', jac=True,
            options={'disp': self.verbose, 'maxiter': self.max_iter},
            tol=self.tol)

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

    def _fit_riemannian(
            self, X, y, weights=None, compute_training_score=False):
        shape = (
            y.shape[-1:] if self.space.default_point_type == 'vector' else
            y.shape[-2:])
        if hasattr(self.metric, 'parallel_transport'):
            def vector_transport(tan_a, tan_b, base_point, _):
                return self.metric.parallel_transport(tan_a, tan_b, base_point)
        else:
            def vector_transport(tan_a, _, __, point):
                return self.space.to_tangent(tan_a, point)

        objective_with_grad = gs.autodiff.value_and_grad(
            lambda params: self._loss(X, y, params, shape, weights))

        lr = self.learning_rate
        intercept_hat = intercept_hat_new = y[0]
        coef_hat = coef_hat_new = self.space.to_tangent(
            gs.random.normal(size=shape), intercept_hat)
        param = gs.vstack(
            [gs.flatten(intercept_hat), gs.flatten(coef_hat)])
        current_loss = math.inf
        current_iter = 0
        for i in range(self.max_iter):
            print("param", param)
            loss, grad = objective_with_grad(param)
            if gs.any(gs.isnan(grad)):
                break
            print("loss")
            print(loss)
            print("grad", grad)
            if loss > current_loss and i > 0:
                lr /= 2
            else:
                if not current_iter % 5:
                    lr *= 2
                coef_hat = coef_hat_new
                intercept_hat = intercept_hat_new
                current_iter += 1
            if abs(loss - current_loss) < self.tol:
                break

            grad_intercept, grad_coef = gs.split(grad, 2)
            riem_grad_intercept = self.space.to_tangent(
                gs.reshape(grad_intercept, shape), intercept_hat)
            riem_grad_coef = self.space.to_tangent(
                gs.reshape(grad_coef, shape), intercept_hat)

            intercept_hat_new = self.metric.exp(
                - lr * riem_grad_intercept, intercept_hat)
            coef_hat_new = vector_transport(
                coef_hat - lr * riem_grad_coef,
                - lr * riem_grad_intercept, 
                intercept_hat, 
                intercept_hat_new)
            # Hack alert
            coef_hat_new = self.space.to_tangent(
                coef_hat_new, intercept_hat_new)
            print("\n\n in riem, is coef hat new in the right space")
            print(self.space.is_tangent(coef_hat_new, intercept_hat_new))
            print(coef_hat_new)

            param = gs.vstack(
                [gs.flatten(intercept_hat_new), gs.flatten(coef_hat_new)])

            current_loss = loss

        self.intercept_ = self.space.projection(intercept_hat)
        self.coef_ = self.space.to_tangent(coef_hat, self.intercept_)
        print("\n\n is it tanget now?")
        print(self.space.is_tangent(self.coef_, self.intercept_ ))
        print(self.coef_)


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
            times = times - self.mean_

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
