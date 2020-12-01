import os
os.environ['GEOMSTATS_BACKEND'] = 'tensorflow'
from scipy.optimize import minimize
import tensorflow as tf

import geomstats.backend as gs
from geomstats.geometry.grassmannian import Grassmannian, GeneralLinear
from geomstats.learning.frechet_mean import FrechetMean, variance

space = Grassmannian(3, 2)
metric = space.metric
gs.random.seed(0)
p_xy = gs.array([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 0.]])


n_samples = 10
data = gs.random.rand(n_samples)
data -= gs.mean(data)
intercept = space.random_uniform()
beta = space.to_tangent(GeneralLinear(3).random_uniform(), intercept)
target = metric.exp(
    tangent_vec=gs.einsum('...,jk->...jk', data, beta),
    base_point=intercept)
estimator = FrechetMean(metric, verbose=True)
estimator.fit(target)
variance_ = variance(target, estimator.estimate_, metric=metric)


def model(x, base_point, tangent_vec):
    return metric.exp(x[:, None] * tangent_vec[None], base_point)


@tf.custom_gradient
def tf_sqdist(x, y):
    e = metric.squared_dist(x, y)

    def grad(dx):
        grd = 2 * metric.log(y, x)
        return grd, 2 * metric.log(x, y)
    return e, grad


def loss(x, y, parameter):
    p, v = gs.split(parameter, 2)
    base_point = gs.reshape(p, (space.n, ) * 2)
    vec = gs.reshape(v, (space.n, ) * 2)
    base_point = GeneralLinear.to_symmetric(base_point)
    tangent_vec = space.to_tangent(vec, base_point)
    exp = metric.exp(gs.einsum('...,...ij->...ij', x, tangent_vec), base_point)
    lo = 1. / 2. * gs.sum(tf_sqdist(exp, y))
    return lo


parameter_ = gs.concatenate([
    gs.flatten(p_xy),
    gs.flatten(space.to_tangent(gs.random.normal(size=(space.n, ) * 2), p_xy))])
objective_with_grad = gs.autograd.value_and_grad(
    lambda param: loss(data, target, param))

res = minimize(
    objective_with_grad, parameter_, jac=True, method='L-BFGS-B',
    options={'disp': True, 'maxiter': 250})

intercept_hat, beta_hat = gs.split(res.x, 2)
intercept_hat = gs.reshape(intercept_hat, (space.n, ) * 2)
beta_hat = gs.reshape(beta_hat, (space.n, ) * 2)
intercept_hat = GeneralLinear.to_symmetric(intercept_hat)
beta_hat = space.to_tangent(beta_hat, intercept_hat)
mse_intercept = metric.squared_dist(intercept_hat, intercept)
