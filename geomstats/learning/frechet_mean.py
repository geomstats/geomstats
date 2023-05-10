"""Frechet mean.

Lead authors: Nicolas Guigui and Nina Miolane.
"""

import logging
import math

from sklearn.base import BaseEstimator

import geomstats.backend as gs
import geomstats.errors as error
from geomstats.geometry.discrete_curves import ElasticMetric, SRVMetric
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric
from geomstats.geometry.matrices import MatricesMetric
from geomstats.geometry.minkowski import MinkowskiMetric

EPSILON = 1e-4

LINEAR_METRICS = [EuclideanMetric, MatricesMetric, MinkowskiMetric]
ELASTIC_METRICS = [SRVMetric, ElasticMetric]


def _is_metric_in_list(metric, metric_classes):
    for metric_class in metric_classes:
        if isinstance(metric, metric_class):
            return True

    return False


def _is_linear_metric(metric_str):
    return _is_metric_in_list(metric_str, LINEAR_METRICS)


def _is_elastic_metric(metric):
    return _is_metric_in_list(metric, ELASTIC_METRICS)


def _scalarmul(scalar, array):
    return gs.einsum("n,n...->n...", scalar, array)


def _scalarmulsum(scalar, array):
    return gs.einsum("n,n...->...", scalar, array)


def _batchscalarmulsum(array_1, array_2):
    return gs.einsum("ni,ni...->i...", array_1, array_2)


def variance(points, base_point, metric, weights=None):
    """Variance of (weighted) points wrt a base point.

    Parameters
    ----------
    points : array-like, shape=[n_samples, dim]
        Points.
    weights : array-like, shape=[n_samples,]
        Weights associated to the points.
        Optional, default: None.

    Returns
    -------
    var : float
       Weighted variance of the points.
    """
    if weights is None:
        n_points = gs.shape(points)[0]
        weights = gs.ones((n_points,))

    sum_weights = gs.sum(weights)
    sq_dists = metric.squared_dist(base_point, points)
    var = weights * sq_dists

    var = gs.sum(var)
    var /= sum_weights

    return var


def linear_mean(points, weights=None):
    """Compute the weighted linear mean.

    The linear mean is the Frechet mean when points:

    - lie in a Euclidean space with Euclidean metric,
    - lie in a Minkowski space with Minkowski metric.

    Parameters
    ----------
    points : array-like, shape=[n_samples, dim]
        Points to be averaged.
    weights : array-like, shape=[n_samples,]
        Weights associated to the points.
        Optional, default: None.

    Returns
    -------
    mean : array-like, shape=[dim,]
        Weighted linear mean of the points.
    """
    if weights is None:
        n_points = gs.shape(points)[0]
        weights = gs.ones(n_points)
    sum_weights = gs.sum(weights)

    weighted_points = _scalarmul(weights, points)

    mean = gs.sum(weighted_points, axis=0) / sum_weights
    return mean


def elastic_mean(points, weights=None, metric=None):
    """Compute the weighted mean of elastic curves.

    SRV: Square Root Velocity.

    SRV curves are a special case of Elastic curves.

    The computation of the mean goes as follows:

    - Transform the curves into their SRVs/F-transform representations,
    - Compute the linear mean of the SRVs/F-transform representations,
    - Inverse-transform the mean in curve space.

    Parameters
    ----------
    points : array-like, shape=[n_samples, k_sampling_points, dim]
        Points on the manifold of curves (i.e. curves) to be averaged.
    weights : array-like, shape=[n_samples,]
        Weights associated to the points (i.e. curves).
        Optional, default: None.

    Returns
    -------
    mean : array-like, shape=[k_sampling_points, dim]
        Weighted linear mean of the points (i.e. of the curves).
    """
    if isinstance(points, list):
        points = gs.stack(points, axis=0)

    transformed = metric.f_transform(points)

    transformed_linear_mean = linear_mean(transformed, weights=weights)

    starting_sampling_point = (
        FrechetMean(metric._space.ambient_manifold.metric)
        .fit(points[:, 0, :], weights=weights)
        .estimate_
    )
    starting_sampling_point = gs.expand_dims(starting_sampling_point, axis=0)
    mean = metric.f_transform_inverse(
        transformed_linear_mean, starting_sampling_point=starting_sampling_point
    )
    return mean


def _default_gradient_descent(
    points,
    metric,
    weights,
    max_iter,
    epsilon,
    init_step_size,
    verbose,
    init_point=None,
):
    """Perform default gradient descent."""
    n_points = gs.shape(points)[0]
    if weights is None:
        weights = gs.ones((n_points,))

    mean = points[0] if init_point is None else init_point

    if n_points == 1:
        return mean

    sum_weights = gs.sum(weights)
    sq_dists_between_iterates = []
    iteration = 0
    sq_dist = 0.0
    var = 0.0

    norm_old = gs.linalg.norm(points)
    step = init_step_size

    while iteration < max_iter:
        logs = metric.log(point=points, base_point=mean)

        var = gs.sum(metric.squared_norm(logs, mean) * weights) / sum_weights

        tangent_mean = _scalarmulsum(weights, logs)
        tangent_mean /= sum_weights
        norm = gs.linalg.norm(tangent_mean)

        sq_dist = metric.squared_norm(tangent_mean, mean)
        sq_dists_between_iterates.append(sq_dist)

        var_is_0 = gs.isclose(var, 0.0)

        metric_dim = metric._space.dim
        if isinstance(metric, ElasticMetric):
            metric_dim = tangent_mean.shape[-2] * tangent_mean.shape[-1]

        sq_dist_is_small = gs.less_equal(sq_dist, epsilon * metric_dim)

        condition = ~gs.logical_or(var_is_0, sq_dist_is_small)
        if not (condition or iteration == 0):
            break

        estimate_next = metric.exp(step * tangent_mean, mean)
        mean = estimate_next
        iteration += 1

        if norm < norm_old:
            norm_old = norm
        elif norm > norm_old:
            step = step / 2.0

    if iteration == max_iter:
        logging.warning(
            "Maximum number of iterations %d reached. The mean may be inaccurate",
            max_iter,
        )

    if verbose:
        logging.info(
            "n_iter: {}, final variance: {}, final dist: {}".format(
                iteration, var, sq_dist
            )
        )

    return mean


def _batch_gradient_descent(
    points,
    metric,
    weights=None,
    max_iter=32,
    init_step_size=1e-3,
    epsilon=5e-3,
    verbose=False,
    init_point=None,
):
    """Perform batch gradient descent."""
    shape = points.shape
    n_points, n_batch = shape[:2]
    point_shape = shape[2:]

    if n_points == 1:
        return points[0]

    if weights is None:
        weights = gs.ones((n_points, n_batch))

    flat_shape = (n_batch * n_points,) + point_shape
    estimates = points[0] if init_point is None else init_point
    points_flattened = gs.reshape(points, (n_points * n_batch,) + point_shape)
    convergence = math.inf
    iteration = 0
    convergence_old = convergence

    while convergence > epsilon and max_iter > iteration:

        iteration += 1
        estimates_broadcast, _ = gs.broadcast_arrays(estimates, points)
        estimates_flattened = gs.reshape(estimates_broadcast, flat_shape)

        tangent_grad = metric.log(points_flattened, estimates_flattened)
        tangent_grad = gs.reshape(tangent_grad, shape)

        tangent_mean = _batchscalarmulsum(weights, tangent_grad) / n_points

        next_estimates = metric.exp(init_step_size * tangent_mean, estimates)
        convergence = gs.sum(metric.squared_norm(tangent_mean, estimates))
        estimates = next_estimates

        if convergence < convergence_old:
            convergence_old = convergence
        elif convergence > convergence_old:
            init_step_size = init_step_size / 2.0

    if iteration == max_iter:
        logging.warning(
            "Maximum number of iterations %d reached. The mean may be inaccurate",
            max_iter,
        )

    if verbose:
        logging.info(
            "n_iter: %d, final dist: %e, final step size: %e",
            iteration,
            convergence,
            init_step_size,
        )

    return estimates


def _adaptive_gradient_descent(
    points,
    metric,
    weights=None,
    max_iter=32,
    epsilon=1e-12,
    init_step_size=1.0,
    init_point=None,
    verbose=False,
):
    """Perform adaptive gradient descent.

    Frechet mean of (weighted) points using adaptive time-steps
    The loss function optimized is :math:`||M_1(x)||_x`
    (where :math:`M_1(x)` is the tangent mean at x) rather than
    the mean-square-distance (MSD) because this simplifies computations.
    Adaptivity is done in a Levenberg-Marquardt style weighting variable tau
    between the first order and the second order Gauss-Newton gradient descent.

    Parameters
    ----------
    points : array-like, shape=[n_samples, *metric.shape]
        Points to be averaged.
    weights : array-like, shape=[n_samples,], optional
        Weights associated to the points.
    max_iter : int, optional
        Maximum number of iterations for the gradient descent.
    init_point : array-like, shape=[*metric.shape]
        Initial point.
        Optional, default : None. In this case the first sample of the input
        data is used.
    epsilon : float, optional
        Tolerance for stopping the gradient descent.

    Returns
    -------
    current_mean: array-like, shape=[*metric.shape]
        Weighted Frechet mean of the points.
    """
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

    tau = init_step_size
    iteration = 0

    logs = metric.log(point=points, base_point=current_mean)
    var = gs.sum(metric.squared_norm(logs, current_mean) * weights) / sum_weights

    current_tangent_mean = _scalarmulsum(weights, logs)
    current_tangent_mean /= sum_weights
    sq_norm_current_tangent_mean = metric.squared_norm(
        current_tangent_mean, base_point=current_mean
    )

    while sq_norm_current_tangent_mean > epsilon**2 and iteration < max_iter:
        iteration += 1

        shooting_vector = tau * current_tangent_mean
        next_mean = metric.exp(tangent_vec=shooting_vector, base_point=current_mean)

        logs = metric.log(point=points, base_point=next_mean)
        var = gs.sum(metric.squared_norm(logs, current_mean) * weights) / sum_weights

        next_tangent_mean = _scalarmulsum(weights, logs)
        next_tangent_mean /= sum_weights
        sq_norm_next_tangent_mean = metric.squared_norm(
            next_tangent_mean, base_point=next_mean
        )

        if sq_norm_next_tangent_mean < sq_norm_current_tangent_mean:
            current_mean = next_mean
            current_tangent_mean = next_tangent_mean
            sq_norm_current_tangent_mean = sq_norm_next_tangent_mean
            tau = min(tau_max, tau_mul_up * tau)
        else:
            tau = max(tau_min, tau_mul_down * tau)

    if iteration == max_iter:
        logging.warning(
            "Maximum number of iterations %d reached. The mean may be inaccurate",
            max_iter,
        )

    if verbose:
        logging.info(
            "n_iter: %d, final variance: %e, final dist: %e, final_step_size: %e",
            iteration,
            var,
            sq_norm_current_tangent_mean,
            tau,
        )

    return current_mean


def _circle_mean(points):
    """Determine the mean on a circle.

    Data are expected in radians in the range [-pi, pi). The mean is returned
    in the same range. If the mean is unique, this algorithm is guaranteed to
    find it. It is not vulnerable to local minima of the Frechet function. If
    the mean is not unique, the algorithm only returns one of the means. Which
    mean is returned depends on numerical rounding errors.

    Parameters
    ----------
    points : array-like, shape=[n_samples, 1]
        Data set of angles (intrinsic coordinates).

    Reference
    ---------
    .. [HH15] Hotz, T. and S. F. Huckemann (2015), "Intrinsic means on the
        circle: Uniqueness, locus and asymptotics", Annals of the Institute of
        Statistical Mathematics 67 (1), 177–193.
        https://arxiv.org/abs/1108.2141
    """
    sample_size = points.shape[0]
    mean0 = gs.mean(points)
    var0 = gs.sum((points - mean0) ** 2)
    sorted_points = gs.sort(points, axis=0)
    means = _circle_variances(mean0, var0, sample_size, sorted_points)
    return means[gs.argmin(means[:, 1]), 0]


def _circle_variances(mean, var, n_samples, points):
    """Compute the minimizer of the variance functional.

    Parameters
    ----------
    mean : float
        Mean angle.
    var : float
        Variance of the angles.
    n_samples : int
        Number of samples.
    points : array-like, shape=[n_samples,]
        Data set of ordered angles.

    References
    ----------
    .. [HH15] Hotz, T. and S. F. Huckemann (2015), "Intrinsic means on the
        circle: Uniqueness, locus and asymptotics", Annals of the Institute of
        Statistical Mathematics 67 (1), 177–193.
        https://arxiv.org/abs/1108.2141
    """
    means = (mean + gs.linspace(0.0, 2 * gs.pi, n_samples + 1)[:-1]) % (2 * gs.pi)
    means = gs.where(means >= gs.pi, means - 2 * gs.pi, means)
    parts = gs.array([gs.sum(points) / n_samples if means[0] < 0 else 0])
    m_plus = means >= 0
    left_sums = gs.cumsum(points)
    right_sums = left_sums[-1] - left_sums
    i = gs.arange(n_samples, dtype=right_sums.dtype)
    j = i[1:]
    parts2 = right_sums[:-1] / (n_samples - j)
    first_term = parts2[:1]
    parts2 = gs.where(m_plus[1:], left_sums[:-1] / j, parts2)
    parts = gs.concatenate([parts, first_term, parts2[1:]])

    # Formula (6) from [HH15]_
    plus_vec = (4 * gs.pi * i / n_samples) * (gs.pi + parts - mean) - (
        2 * gs.pi * i / n_samples
    ) ** 2
    minus_vec = (4 * gs.pi * (n_samples - i) / n_samples) * (gs.pi - parts + mean) - (
        2 * gs.pi * (n_samples - i) / n_samples
    ) ** 2
    minus_vec = gs.where(m_plus, plus_vec, minus_vec)
    means = gs.transpose(gs.vstack([means, var + minus_vec]))
    return means


class FrechetMean(BaseEstimator):
    r"""Empirical Frechet mean.

    Parameters
    ----------
    metric : RiemannianMetric
        Riemannian metric.
    max_iter : int
        Maximum number of iterations for gradient descent.
        Optional, default: 32.
    epsilon : float
        Tolerance for stopping the gradient descent.
        Optional, default : 1e-4
    method : str, {\'default\', \'adaptive\', \'batch\'}
        Gradient descent method.
        The `adaptive` method uses a Levenberg-Marquardt style adaptation of
        the learning rate. The `batch` method is similar to the default
        method but for batches of equal length of samples. In this case,
        samples must be of shape [n_samples, n_batch, *metric.shape].
        Optional, default: \'default\'.
    init_point : array-like, shape=[*metric.shape]
        Initial point.
        Optional, default : None. In this case the first sample of the input
        data is used.
    init_step_size : float
        Initial step size or learning rate.
    verbose : bool
        Verbose option.
        Optional, default: False.

    Attributes
    ----------
    estimate_ : array-like, shape=[*metric.shape]
        If fit, Frechet mean.
    """

    def __init__(
        self,
        metric,
        max_iter=32,
        epsilon=EPSILON,
        method="default",
        init_point=None,
        init_step_size=1.0,
        verbose=False,
    ):
        self.metric = metric

        self.method = method

        self.max_iter = max_iter
        self.epsilon = epsilon
        self.init_step_size = init_step_size
        self.verbose = verbose
        self.init_point = init_point
        self.estimate_ = None

    @property
    def method(self):
        """Gradient descent method."""
        return self._method

    @method.setter
    def method(self, value):
        """Gradient descent method."""
        error.check_parameter_accepted_values(
            value, "method", ["default", "adaptive", "batch"]
        )
        self._method = value

    @property
    def _minimize(self):
        MAP_OPTIMIZER = {
            "default": _default_gradient_descent,
            "adaptive": _adaptive_gradient_descent,
            "batch": _batch_gradient_descent,
        }
        minimize_ = MAP_OPTIMIZER.get(self.method)
        return lambda points, weights, metric: minimize_(
            points=points,
            weights=weights,
            metric=metric,
            max_iter=self.max_iter,
            init_step_size=self.init_step_size,
            epsilon=self.epsilon,
            verbose=self.verbose,
            init_point=self.init_point,
        )

    def fit(self, X, y=None, weights=None):
        """Compute the empirical weighted Frechet mean.

        Parameters
        ----------
        X : array-like, shape=[n_samples, *metric.shape]
            Training input samples.
        y : None
            Target values. Ignored.
        weights : array-like, shape=[n_samples,]
            Weights associated to the samples.
            Optional, default: None, in which case it is equally weighted.

        Returns
        -------
        self : object
            Returns self.
        """
        if isinstance(self.metric, HypersphereMetric) and self.metric._space.dim == 1:
            mean = Hypersphere.angle_to_extrinsic(
                _circle_mean(Hypersphere.extrinsic_to_angle(X))
            )

        elif _is_linear_metric(self.metric):
            mean = linear_mean(points=X, weights=weights)

        elif _is_elastic_metric(self.metric):
            mean = elastic_mean(points=X, weights=weights, metric=self.metric)

        else:
            mean = self._minimize(
                points=X,
                weights=weights,
                metric=self.metric,
            )

        self.estimate_ = mean

        return self
