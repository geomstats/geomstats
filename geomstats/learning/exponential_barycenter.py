"""Frechet mean."""

import logging

from sklearn.base import BaseEstimator

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean

EPSILON = 1e-6


def linear_mean(points, weights=None):
    """Compute the weighted linear mean.

    The linear mean is the Frechet mean when points:
    - lie in a Euclidean space with Euclidean metric,
    - lie in a Minkowski space with Minkowski metric.

    Parameters
    ----------
    points : array-like, shape=[n_samples, dimension]
        Points to be averaged.

    weights : array-like, shape=[n_samples, 1], optional
        Weights associated to the points.

    Returns
    -------
    mean : array-like, shape=[1, dimension]
        Weighted linear mean of the points.
    """
    if isinstance(points, list):
        points = gs.vstack(points)
    points = gs.to_ndarray(points, to_ndim=2)
    n_points = gs.shape(points)[0]

    if isinstance(weights, list):
        weights = gs.vstack(weights)
    elif weights is None:
        weights = gs.ones((n_points, ))

    weighted_points = gs.einsum('...,...j->...j', weights, points)
    mean = (gs.sum(weighted_points, axis=0) / gs.sum(weights))
    mean = gs.to_ndarray(mean, to_ndim=2)
    return mean


def _default_gradient_descent(
        group, points, weights=None, max_iter=32, step=1.,
        epsilon=EPSILON, verbose=False):
    """Compute the (weighted) group exponential barycenter of `points`.

    Parameters
    ----------
    group : LieGroup
        Instance of the class LieGroup.
    points : array-like, shape=[n_samples, [n,n]]
        Input points lying in the Lie Group.
    weights : array-like, shape=[n_samples,]
        default is 1 for each point
        Weights of each point.
    max_iter : int, optional (defaults to 32)
        The maximum number of iterations to perform in the gradient descent.
    epsilon : float, optional (defaults to 1e-6)
        The tolerance to reach convergence. The exstrinsic norm of the
        gradient is used as criterion.
    step : float, optional (defaults to 1.)
        The learning rate in the gradient descent.
    verbose : bool
        Level of verbosity to inform about convergence.

    Returns
    -------
    exp_bar : array-like, shape=[n,n]
        The exponential_barycenter of the input points.
    """
    n_points = points.shape[0]
    if weights is None:
        weights = gs.ones((n_points,))
    sum_weights = gs.sum(weights)

    def while_loop_cond(iter_index, current, grad_norm):
        result = grad_norm > epsilon
        return result or iter_index == 0

    def while_loop_body(iter_index, current, grad_norm):
        logs = group.log(point=points, base_point=current)
        tangent_mean = step * gs.einsum(
            'n, nkl->kl', weights / sum_weights, logs)
        mean_next = group.exp(tangent_vec=tangent_mean, base_point=current)

        grad_norm = gs.linalg.norm(tangent_mean)
        sq_dists_between_iterates.append(grad_norm)

        current = mean_next
        iter_index += 1
        return [iter_index, current, grad_norm]

    mean = points[0]
    if n_points == 1:
        return mean

    mean = gs.to_ndarray(mean, to_ndim=3)

    sq_dists_between_iterates = []
    iteration = 0
    norm = 0.

    last_iteration, mean, norm = gs.while_loop(
        lambda i, m, sq: while_loop_cond(i, m, sq),
        lambda i, m, sq: while_loop_body(i, m, sq),
        loop_vars=[iteration, mean, norm],
        maximum_iterations=max_iter)

    if last_iteration == max_iter:
        logging.warning(
            'Maximum number of iterations {} reached. '
            'The mean may be inaccurate'.format(max_iter))

    if verbose:
        logging.info('n_iter: {}, final norm: {}'.format(last_iteration, norm))

    return mean[0]


class ExponentialBarycenter(BaseEstimator):
    """Empirical Exponential Barycenter for Matrix groups.

    Parameters
    ----------
    group : LieGroup
        A Lie group instance on which the data lie.
    max_iter : int, optional (defaults to 32)
        The maximum number of iterations to perform in the gradient descent.
    epsilon : float, optional (defaults to 1e-6)
        The tolerance to reach convergence. The exstrinsic norm of the
        gradient is used as criterion.
    step : float, optional (defaults to 1.)
        The learning rate in the gradient descent.
    verbose : bool
        Level of verbosity to inform about convergence.

    Attributes
    ----------
    estimate_ : array-like, shape=[dimension, dimension]
    """

    def __init__(self, group,
                 max_iter=32,
                 epsilon=EPSILON,
                 step=1.,
                 verbose=False):
        self.group = group
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.verbose = verbose
        self.step = step
        self.estimate_ = None

    def fit(self, X, y=None, weights=None):
        """Compute the empirical Exponential Barycenter mean.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
            Ignored
        weights : array-like, shape=[n_samples,], optional

        Returns
        -------
        self : object
            Returns self.
        """
        if isinstance(self.group, Euclidean):
            mean = linear_mean(points=X, weights=weights)

        else:
            mean = _default_gradient_descent(
                points=X, weights=weights, group=self.group,
                max_iter=self.max_iter,
                epsilon=self.epsilon,
                step=self.step,
                verbose=self.verbose)

        self.estimate_ = mean

        return self
