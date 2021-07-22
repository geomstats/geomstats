"""Frechet mean."""

import logging
import math

from sklearn.base import BaseEstimator

import geomstats.backend as gs
import geomstats.errors as error
import geomstats.vectorization

EPSILON = 1e-4


def variance(points,
             base_point,
             metric,
             weights=None,
             point_type='vector'):
    """Variance of (weighted) points wrt a base point.

    Parameters
    ----------
    points : array-like, shape=[..., dim]
        Points.
    weights : array-like, shape=[...,]
        Weights associated to the points.
        Optional, default: None.

    Returns
    -------
    var : float
       Weighted variance of the points.
    """
    n_points = geomstats.vectorization.get_n_points(
        points, point_type)

    if weights is None:
        weights = gs.ones((n_points,))

    sum_weights = gs.sum(weights)
    sq_dists = metric.squared_dist(base_point, points)
    var = weights * sq_dists

    var = gs.sum(var)
    var /= sum_weights

    return var


def linear_mean(points, weights=None, point_type='vector'):
    """Compute the weighted linear mean.

    The linear mean is the Frechet mean when points:
    - lie in a Euclidean space with Euclidean metric,
    - lie in a Minkowski space with Minkowski metric.

    Parameters
    ----------
    points : array-like, shape=[..., dim]
        Points to be averaged.
    weights : array-like, shape=[...,]
        Weights associated to the points.
        Optional, default: None.

    Returns
    -------
    mean : array-like, shape=[dim,]
        Weighted linear mean of the points.
    """
    if isinstance(points, list):
        points = gs.stack(points, axis=0)
    if isinstance(weights, list):
        weights = gs.array(weights)

    n_points = geomstats.vectorization.get_n_points(
        points, point_type)

    if weights is None:
        weights = gs.ones((n_points,))
    sum_weights = gs.sum(weights)

    einsum_str = '...,...j->...j'
    if point_type == 'matrix':
        einsum_str = '...,...jk->...jk'

    weighted_points = gs.einsum(einsum_str, weights, points)

    mean = gs.sum(weighted_points, axis=0) / sum_weights
    return mean


def _default_gradient_descent(
        points, metric, weights, max_iter, point_type, epsilon,
        initial_step_size, verbose):
    """Perform default gradient descent."""
    if point_type == 'vector':
        points = gs.to_ndarray(points, to_ndim=2)
        einsum_str = 'n,nj->j'
    else:
        points = gs.to_ndarray(points, to_ndim=3)
        einsum_str = 'n,nij->ij'
    n_points = gs.shape(points)[0]

    if weights is None:
        weights = gs.ones((n_points,))

    mean = points[0]

    if n_points == 1:
        return mean

    sum_weights = gs.sum(weights)
    sq_dists_between_iterates = []
    iteration = 0
    sq_dist = 0.
    var = 0.

    norm_old = gs.linalg.norm(points)
    step = initial_step_size

    while iteration < max_iter:
        logs = metric.log(point=points, base_point=mean)

        var = gs.sum(
            metric.squared_norm(logs, mean) * weights) / gs.sum(weights)

        tangent_mean = gs.einsum(einsum_str, weights, logs)
        tangent_mean /= sum_weights
        norm = gs.linalg.norm(tangent_mean)

        sq_dist = metric.squared_norm(tangent_mean, mean)
        sq_dists_between_iterates.append(sq_dist)

        var_is_0 = gs.isclose(var, 0.)
        sq_dist_is_small = gs.less_equal(sq_dist, epsilon * metric.dim)
        condition = ~gs.logical_or(var_is_0, sq_dist_is_small)
        if not (condition or iteration == 0):
            break

        estimate_next = metric.exp(step * tangent_mean, mean)
        mean = estimate_next
        iteration += 1

        if norm < norm_old:
            norm_old = norm
        elif norm > norm_old:
            step = step / 2.

    if iteration == max_iter:
        logging.warning(
            'Maximum number of iterations {} reached. '
            'The mean may be inaccurate'.format(max_iter))

    if verbose:
        logging.info('n_iter: {}, final variance: {}, final dist: {}'.format(
            iteration, var, sq_dist))

    return mean


def _batch_gradient_descent(
        points, metric, weights=None, max_iter=32, lr=1e-3, epsilon=5e-3,
        point_type='vector', verbose=False):
    """Perform batch gradient descent."""
    if point_type == 'vector':
        if points.ndim < 3:
            return _default_gradient_descent(
                points, metric, weights, max_iter, point_type, epsilon,
                lr, verbose)
        einsum_str = 'ni,nij->ij'
        ndim = 1
    else:
        if points.ndim < 4:
            return _default_gradient_descent(
                points, metric, weights, max_iter, point_type, epsilon,
                lr, verbose)
        einsum_str = 'nk,nkij->kij'
        ndim = 2

    shape = points.shape
    n_points = shape[0]
    n_batch = shape[1]

    if n_points == 1:
        return points[0]

    if weights is None:
        weights = gs.ones((n_points, n_batch))

    flat_shape = (n_batch * n_points, ) + shape[-ndim:]
    estimates = points[0]
    points_flattened = gs.reshape(
        points, (n_points * n_batch, ) + shape[-ndim:])
    convergence = math.inf
    iteration = 0
    convergence_old = convergence

    while convergence > epsilon and max_iter > iteration:

        iteration += 1
        estimates_broadcast, _ = gs.broadcast_arrays(estimates, points)
        estimates_flattened = gs.reshape(estimates_broadcast, flat_shape)

        tangent_grad = metric.log(points_flattened, estimates_flattened)
        tangent_grad = gs.reshape(tangent_grad, shape)

        tangent_mean = gs.einsum(einsum_str, weights, tangent_grad) / n_points

        next_estimates = metric.exp(lr * tangent_mean, estimates)
        convergence = gs.sum(metric.squared_norm(tangent_mean, estimates))
        estimates = next_estimates

        if convergence < convergence_old:
            convergence_old = convergence
        elif convergence > convergence_old:
            lr = lr / 2.

    if iteration == max_iter:
        logging.warning(
            'Maximum number of iterations {} reached. The '
            'mean may be inaccurate'.format(max_iter))

    if verbose:
        logging.info(
            'n_iter: {}, final dist: {},'
            'final step size: {}'.format(iteration, convergence, lr))

    return estimates


def _adaptive_gradient_descent(
        points, metric, weights=None, max_iter=32, epsilon=1e-12,
        initial_tau=1., init_point=None, point_type='vector', verbose=False):
    """Perform adaptive gradient descent.

    Frechet mean of (weighted) points using adaptive time-steps
    The loss function optimized is :math:`||M_1(x)||_x`
    (where :math:`M_1(x)` is the tangent mean at x) rather than
    the mean-square-distance (MSD) because this simplifies computations.
    Adaptivity is done in a Levenberg-Marquardt style weighting variable tau
    between the first order and the second order Gauss-Newton gradient descent.

    Parameters
    ----------
    points : array-like, shape=[..., dim]
        Points to be averaged.
    weights : array-like, shape=[..., 1], optional
        Weights associated to the points.
    max_iter : int, optional
        Maximum number of iterations for the gradient descent.
    init_point : array-like, shape=[n_init, dimension], optional
        Initial point.
    epsilon : float, optional
        Tolerance for stopping the gradient descent.

    Returns
    -------
    current_mean: array-like, shape=[..., dim]
        Weighted Frechet mean of the points.
    """
    if point_type == 'vector':
        points = gs.to_ndarray(points, to_ndim=2)
        einsum_str = 'n,nj->j'
    else:
        points = gs.to_ndarray(points, to_ndim=3)
        einsum_str = 'n,nij->ij'
    n_points = gs.shape(points)[0]

    tau_max = 1e6
    tau_mul_up = 1.6511111
    tau_min = 1e-6
    tau_mul_down = 0.1

    if n_points == 1:
        return points[0]

    current_mean = points[0] if init_point is None else init_point

    if weights is None:
        weights = gs.ones((n_points,))
    sum_weights = gs.sum(weights)

    tau = initial_tau
    iteration = 0

    logs = metric.log(point=points, base_point=current_mean)
    var = gs.sum(
        metric.squared_norm(logs, current_mean) * weights
    ) / gs.sum(weights)

    current_tangent_mean = gs.einsum(einsum_str, weights, logs)
    current_tangent_mean /= sum_weights
    sq_norm_current_tangent_mean = metric.squared_norm(
        current_tangent_mean, base_point=current_mean)

    while (sq_norm_current_tangent_mean > epsilon ** 2
           and iteration < max_iter):
        iteration += 1

        shooting_vector = tau * current_tangent_mean
        next_mean = metric.exp(
            tangent_vec=shooting_vector, base_point=current_mean)

        logs = metric.log(point=points, base_point=next_mean)
        var = gs.sum(
            metric.squared_norm(logs, current_mean) * weights
        ) / gs.sum(weights)

        next_tangent_mean = gs.einsum(einsum_str, weights, logs)
        next_tangent_mean /= sum_weights
        sq_norm_next_tangent_mean = metric.squared_norm(
            next_tangent_mean, base_point=next_mean)

        if sq_norm_next_tangent_mean < sq_norm_current_tangent_mean:
            current_mean = next_mean
            current_tangent_mean = next_tangent_mean
            sq_norm_current_tangent_mean = sq_norm_next_tangent_mean
            tau = min(tau_max, tau_mul_up * tau)
        else:
            tau = max(tau_min, tau_mul_down * tau)

    if iteration == max_iter:
        logging.warning(
            'Maximum number of iterations {} reached. '
            'The mean may be inaccurate'.format(max_iter))

    if verbose:
        logging.info(
            'n_iter: {}, final variance: {}, final dist: {},'
            ' final_step_size: {}'.format(
                iteration, var, sq_norm_current_tangent_mean, tau))

    return current_mean


class FrechetMean(BaseEstimator):
    r"""Empirical Frechet mean.

    Parameters
    ----------
    metric : RiemannianMetric
        Riemannian metric.
    max_iter : int
        Maximum number of iterations for gradient descent.
        Optional, default: 32.
    point_type : str, {\'vector\', \'matrix\'}
        Point type.
        Optional, default: None.
    method : str, {\'default\', \'adaptive\', \'batch\'}
        Gradient descent method.
        The `adaptive` method uses a Levenberg-Marquardt style adaptation of
        the learning rate. The `batch` method is similar to the default
        method but for batches of equal length of samples. In this case,
        samples must be of shape [n_samples, n_batch, {dim, [n,n]}].
        Optional, default: \'default\'.
    verbose : bool
        Verbose option.
        Optional, default: False.
    """

    def __init__(self, metric,
                 max_iter=32,
                 epsilon=EPSILON,
                 point_type=None,
                 method='default',
                 lr=1.,
                 verbose=False):

        self.metric = metric
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.point_type = point_type
        self.method = method
        self.lr = lr
        self.verbose = verbose
        self.estimate_ = None

        if point_type is None:
            self.point_type = metric.default_point_type
        error.check_parameter_accepted_values(
            self.point_type, 'point_type', ['vector', 'matrix'])

    def fit(self, X, y=None, weights=None):
        """Compute the empirical Frechet mean.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[..., {dim, [n, n]}]
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
        metric_str = self.metric.__str__()
        is_linear_metric = (
            'EuclideanMetric' in metric_str
            or 'MatricesMetric' in metric_str
            or 'MinkowskiMetric' in metric_str)

        error.check_parameter_accepted_values(
            self.method, 'method', ['default', 'adaptive', 'batch'])

        if is_linear_metric:
            mean = linear_mean(
                points=X, weights=weights, point_type=self.point_type)

        elif self.method == 'default':
            mean = _default_gradient_descent(
                points=X, weights=weights, metric=self.metric,
                max_iter=self.max_iter, initial_step_size=self.lr,
                point_type=self.point_type, epsilon=self.epsilon,
                verbose=self.verbose)
        elif self.method == 'adaptive':
            mean = _adaptive_gradient_descent(
                points=X, weights=weights, metric=self.metric,
                max_iter=self.max_iter, point_type=self.point_type,
                epsilon=self.epsilon, verbose=self.verbose,
                initial_tau=self.lr)
        elif self.method == 'batch':
            mean = _batch_gradient_descent(
                points=X, weights=weights, metric=self.metric,
                lr=self.lr, epsilon=self.epsilon, max_iter=self.max_iter,
                point_type=self.point_type, verbose=self.verbose)

        self.estimate_ = mean

        return self
