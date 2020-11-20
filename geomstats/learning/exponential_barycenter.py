"""Exponential barycenter."""

import logging

from sklearn.base import BaseEstimator

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.learning.frechet_mean import linear_mean

EPSILON = 1e-6


def _default_gradient_descent(
        group, points, weights=None, max_iter=32, step=1.,
        epsilon=EPSILON, verbose=False):
    """Compute the (weighted) group exponential barycenter of `points`.

    Parameters
    ----------
    group : LieGroup
        Instance of the class LieGroup.
    points : array-like, shape=[..., [n,n]]
        Input points lying in the Lie Group.
    weights : array-like, shape=[...,]
        Weights associated to the points.
        Optional, defaults to 1 for each point if None.
    max_iter : int
        Maximum number of iterations to perform in the gradient descent.
        Optional, default: 32.
    epsilon : float
        Tolerance to reach convergence. The exstrinsic norm of the
        gradient is used as criterion.
        Optional, default: 1e-6.
    step : float
        Learning rate in the gradient descent.
        Optional, default: 1.
    verbose : bool
        Level of verbosity to inform about convergence.
        Optional, default: False.

    Returns
    -------
    exp_bar : array-like, shape=[n,n]
        Exponential_barycenter of the input points.
    """
    ndim = 2 if group.default_point_type == 'vector' else 3
    if gs.ndim(gs.array(points)) < ndim or len(points) == 1:
        return points[0] if len(points) == 1 else points

    n_points = points.shape[0]
    if weights is None:
        weights = gs.ones((n_points,))
    weights = gs.cast(weights, gs.float32)
    sum_weights = gs.sum(weights)

    mean = points[0]

    sq_dists_between_iterates = []
    iteration = 0
    grad_norm = 0.

    while iteration < max_iter:
        if not (grad_norm > epsilon or iteration == 0):
            break
        inv_mean = group.inverse(mean)
        centered_points = group.compose(inv_mean, points)
        logs = group.log(point=centered_points)
        tangent_mean = step * gs.einsum(
            'n, nk...->k...', weights / sum_weights, logs)
        mean_next = group.compose(
            mean,
            group.exp(tangent_vec=tangent_mean))

        grad_norm = gs.linalg.norm(tangent_mean)
        sq_dists_between_iterates.append(grad_norm)

        mean = mean_next
        iteration += 1

    if iteration == max_iter:
        logging.warning(
            'Maximum number of iterations {} reached. '
            'The mean may be inaccurate'.format(max_iter))

    if verbose:
        logging.info(
            'n_iter: {}, final gradient norm: {}'.format(iteration, grad_norm))
    return mean


class ExponentialBarycenter(BaseEstimator):
    """Empirical Exponential Barycenter for Matrix groups.

    Parameters
    ----------
    group : LieGroup
        Lie group instance on which the data lie.
    max_iter : int
        Maximum number of iterations to perform in the gradient descent.
        Optional, default: 32.
    epsilon : float
        Tolerance to reach convergence. The exstrinsic norm of the
        gradient is used as criterion.
        Optional, default: 1e-6.
    step : float
        Learning rate in the gradient descent.
        Optional, default: 1.
    verbose : bool
        Level of verbosity to inform about convergence.
        Optional, default: 1.

    Attributes
    ----------
    estimate_ : array-like, shape=[dim, dim]
    """

    def __init__(self, group,
                 max_iter=32,
                 epsilon=EPSILON,
                 step=1.,
                 point_type=None,
                 verbose=False):
        self.group = group
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.verbose = verbose
        self.step = step
        self.point_type = point_type
        self.estimate_ = None

    def fit(self, X, y=None, weights=None):
        """Compute the empirical Exponential Barycenter mean.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[..., n_features]
            Training input samples.
        y : array-like, shape=[...,] or [..., n_outputs]
            Target values (class labels in classification, real numbers in
            regression).
            Ignored.
        weights : array-like, shape=[...,]
            Weights associated to the points.
            Optional, default: None.

        Returns
        -------
        self : object
            Returns self.
        """
        if isinstance(self.group, Euclidean):
            mean = linear_mean(points=X, weights=weights)

        # TODO (nguigs): use closed form expression for special euclidean
        #  group as before PR #537

        else:
            mean = _default_gradient_descent(
                points=X, weights=weights, group=self.group,
                max_iter=self.max_iter,
                epsilon=self.epsilon,
                step=self.step,
                verbose=self.verbose)
        self.estimate_ = mean

        return self
