"""Frechet mean.

Lead authors: Nicolas Guigui and Nina Miolane.
"""

import abc
import logging
import math

import numpy as np
from sklearn.base import BaseEstimator

import geomstats.backend as gs
import geomstats.errors as error
from geomstats.geometry.discrete_curves import ElasticMetric, SRVMetric
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.hypersphere import HypersphereMetric
from geomstats.geometry.stratified.bhv_space import BHVMetric

ELASTIC_METRICS = [SRVMetric, ElasticMetric]


def _is_linear_metric(metric):
    return isinstance(metric, EuclideanMetric)


def _is_elastic_metric(metric):
    return isinstance(metric, tuple(ELASTIC_METRICS))


def _is_bhv_metric(metric):
    return isinstance(metric, BHVMetric)


def _scalarmul(scalar, array):
    return gs.einsum("n,n...->n...", scalar, array)


def _scalarmulsum(scalar, array):
    return gs.einsum("n,n...->...", scalar, array)


def _batchscalarmulsum(array_1, array_2):
    return gs.einsum("ni,ni...->i...", array_1, array_2)


def variance(space, points, base_point, weights=None):
    """Variance of (weighted) points wrt a base point.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
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
    sq_dists = space.metric.squared_dist(base_point, points)
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


class BaseGradientDescent(abc.ABC):
    """Base class for gradient descent.

    Parameters
    ----------
    max_iter : int, optional
        Maximum number of iterations for the gradient descent.
    epsilon : float, optional
        Tolerance for stopping the gradient descent.
    init_point : array-like, shape=[*metric.shape]
        Initial point.
        Optional, default : None. In this case the first sample of the input
        data is used.
    init_step_size : float
        Learning rate in the gradient descent.
        Optional, default: 1.
    verbose : bool
        Level of verbosity to inform about convergence.
        Optional, default: False.
    """

    def __init__(
        self,
        max_iter=32,
        epsilon=1e-4,
        init_point=None,
        init_step_size=1.0,
        verbose=False,
    ):
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.init_step_size = init_step_size
        self.verbose = verbose
        self.init_point = init_point

    @abc.abstractmethod
    def minimize(self, space, points, weights=None):
        """Perform gradient descent."""
        pass


class GradientDescent(BaseGradientDescent):
    """Default gradient descent."""

    def minimize(self, space, points, weights=None):
        """Perform default gradient descent."""
        n_points = gs.shape(points)[0]
        if weights is None:
            weights = gs.ones((n_points,))

        mean = points[0] if self.init_point is None else self.init_point

        if n_points == 1:
            return mean

        sum_weights = gs.sum(weights)
        iteration = 0
        sq_dist = 0.0
        var = 0.0

        norm_old = gs.linalg.norm(points)
        step_size = self.init_step_size

        while iteration < self.max_iter:
            logs = space.metric.log(point=points, base_point=mean)

            var = gs.sum(space.metric.squared_norm(logs, mean) * weights) / sum_weights

            tangent_mean = _scalarmulsum(weights, logs)
            tangent_mean /= sum_weights
            norm = gs.linalg.norm(tangent_mean)

            sq_dist = space.metric.squared_norm(tangent_mean, mean)

            var_is_0 = gs.isclose(var, 0.0)

            sq_dist_is_small = gs.less_equal(sq_dist, self.epsilon * space.dim)

            condition = ~gs.logical_or(var_is_0, sq_dist_is_small)
            if not (condition or iteration == 0):
                break

            estimate_next = space.metric.exp(step_size * tangent_mean, mean)
            mean = estimate_next
            iteration += 1

            if norm < norm_old:
                norm_old = norm
            elif norm > norm_old:
                step_size = step_size / 2.0

        if iteration == self.max_iter:
            logging.warning(
                "Maximum number of iterations %d reached. The mean may be inaccurate",
                self.max_iter,
            )

        if self.verbose:
            logging.info(
                "n_iter: {}, final variance: {}, final dist: {}".format(
                    iteration, var, sq_dist
                )
            )

        return mean


class BatchGradientDescent(BaseGradientDescent):
    """Batch gradient descent."""

    def minimize(self, space, points, weights=None):
        """Perform batch gradient descent."""
        shape = points.shape
        n_points, n_batch = shape[:2]
        point_shape = shape[2:]

        if n_points == 1:
            return points[0]

        if weights is None:
            weights = gs.ones((n_points, n_batch))

        flat_shape = (n_batch * n_points,) + point_shape
        estimates = points[0] if self.init_point is None else self.init_point
        points_flattened = gs.reshape(points, (n_points * n_batch,) + point_shape)
        convergence = math.inf
        iteration = 0
        convergence_old = convergence

        step_size = self.init_step_size

        while convergence > self.epsilon and self.max_iter > iteration:
            iteration += 1
            estimates_broadcast, _ = gs.broadcast_arrays(estimates, points)
            estimates_flattened = gs.reshape(estimates_broadcast, flat_shape)

            tangent_grad = space.metric.log(points_flattened, estimates_flattened)
            tangent_grad = gs.reshape(tangent_grad, shape)

            tangent_mean = _batchscalarmulsum(weights, tangent_grad) / n_points

            next_estimates = space.metric.exp(step_size * tangent_mean, estimates)
            convergence = gs.sum(space.metric.squared_norm(tangent_mean, estimates))
            estimates = next_estimates

            if convergence < convergence_old:
                convergence_old = convergence
            elif convergence > convergence_old:
                step_size = step_size / 2.0

        if iteration == self.max_iter:
            logging.warning(
                "Maximum number of iterations %d reached. The mean may be inaccurate",
                self.max_iter,
            )

        if self.verbose:
            logging.info(
                "n_iter: %d, final dist: %e, final step size: %e",
                iteration,
                convergence,
                step_size,
            )

        return estimates


class AdaptiveGradientDescent(BaseGradientDescent):
    """Adaptive gradient descent."""

    def minimize(self, space, points, weights=None):
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

        current_mean = points[0] if self.init_point is None else self.init_point

        if weights is None:
            weights = gs.ones((n_points,))
        sum_weights = gs.sum(weights)

        tau = self.init_step_size
        iteration = 0

        logs = space.metric.log(point=points, base_point=current_mean)
        var = (
            gs.sum(space.metric.squared_norm(logs, current_mean) * weights)
            / sum_weights
        )

        current_tangent_mean = _scalarmulsum(weights, logs)
        current_tangent_mean /= sum_weights
        sq_norm_current_tangent_mean = space.metric.squared_norm(
            current_tangent_mean, base_point=current_mean
        )

        while (
            sq_norm_current_tangent_mean > self.epsilon**2 and iteration < self.max_iter
        ):
            iteration += 1

            shooting_vector = tau * current_tangent_mean
            next_mean = space.metric.exp(
                tangent_vec=shooting_vector, base_point=current_mean
            )

            logs = space.metric.log(point=points, base_point=next_mean)
            var = (
                gs.sum(space.metric.squared_norm(logs, current_mean) * weights)
                / sum_weights
            )

            next_tangent_mean = _scalarmulsum(weights, logs)
            next_tangent_mean /= sum_weights
            sq_norm_next_tangent_mean = space.metric.squared_norm(
                next_tangent_mean, base_point=next_mean
            )

            if sq_norm_next_tangent_mean < sq_norm_current_tangent_mean:
                current_mean = next_mean
                current_tangent_mean = next_tangent_mean
                sq_norm_current_tangent_mean = sq_norm_next_tangent_mean
                tau = min(tau_max, tau_mul_up * tau)
            else:
                tau = max(tau_min, tau_mul_down * tau)

        if iteration == self.max_iter:
            logging.warning(
                "Maximum number of iterations %d reached. The mean may be inaccurate",
                self.max_iter,
            )

        if self.verbose:
            logging.info(
                "n_iter: %d, final variance: %e, final dist: %e, final_step_size: %e",
                iteration,
                var,
                sq_norm_current_tangent_mean,
                tau,
            )

        return current_mean


class LinearMean(BaseEstimator):
    """Linear mean.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.

    Attributes
    ----------
    estimate_ : array-like, shape=[*space.shape]
        If fit, Frechet mean.
    """

    def __init__(self, space):
        self.space = space
        self.estimate_ = None

    def fit(self, X, y=None, weights=None):
        """Compute the Euclidean mean.

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
        self.estimate_ = linear_mean(points=X, weights=weights)
        return self


class ElasticMean(BaseEstimator):
    """Elastic mean.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.

    Attributes
    ----------
    estimate_ : array-like, shape=[*space.shape]
        If fit, Frechet mean.
    """

    def __init__(self, space):
        self.space = space
        self.estimate_ = None

    def _elastic_mean(self, points, weights=None):
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
        diffeo = self.space.metric.diffeo
        transformed = diffeo(points)
        transformed_linear_mean = linear_mean(transformed, weights=weights)

        return diffeo.inverse(transformed_linear_mean)

    def fit(self, X, y=None, weights=None):
        """Compute the elastic mean.

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
        self.estimate_ = self._elastic_mean(X, weights=weights)
        return self


class CircleMean(BaseEstimator):
    """Circle mean.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.

    Attributes
    ----------
    estimate_ : array-like, shape=[2,]
        If fit, Frechet mean.
    """

    def __init__(self, space):
        self.space = space
        self.estimate_ = None

    def _circle_mean(self, points):
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
        means = self._circle_variances(mean0, var0, sample_size, sorted_points)
        return means[gs.argmin(means[:, 1]), 0]

    @staticmethod
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
        minus_vec = (4 * gs.pi * (n_samples - i) / n_samples) * (
            gs.pi - parts + mean
        ) - (2 * gs.pi * (n_samples - i) / n_samples) ** 2
        minus_vec = gs.where(m_plus, plus_vec, minus_vec)
        means = gs.transpose(gs.vstack([means, var + minus_vec]))
        return means

    def fit(self, X, y=None):
        """Compute the circle mean.

        Parameters
        ----------
        X : array-like, shape=[n_samples, 2]
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
        self.estimate_ = self.space.angle_to_extrinsic(
            self._circle_mean(self.space.extrinsic_to_angle(X))
        )
        return self


class SturmsMean(BaseEstimator):
    """Frechet mean using Sturm's algorithm.

    Some geodesic metric spaces (like BHV) do not have an exp, and can
    therefore not use gradient-descent-like optimisation. Sturm's Algorithm
    works by iteratively computing geodesics between data points and an
    updated estimate on the previous geodesic, without log or exp.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    max_iter : int, optional
        Maximum number of iterations.
    epsilon : float, optional
        Tolerance for stopping iterative computation.

    Attributes
    ----------
    estimate_ : array-like, shape=[*space.shape]
        If fit, Frechet mean.
    """

    def __init__(self, space, max_iter=32, epsilon=1e-4):
        self.max_iter = max_iter
        self.epsilon = epsilon

        self.space = space
        self.estimate_ = None

    def fit(self, X, y=None, weights=None):
        """Compute the weighted mean for geodesic metric spaces.

        https://www.iam.uni-bonn.de/fileadmin/WT/Inhalt/people/Karl-Theodor_Sturm/papers/paper41.pdf
        https://pmc.ncbi.nlm.nih.gov/articles/PMC5793493/

        The computation of the mean goes as follows:
            - Initialise mean estimate, i
            - Sample point
            - Compute geodesic between mean estimate and sampled point
            - Compute new mean estimate 1/(i+2) of the way down geodesic
            - Rinse, repeat

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
        # need weights to sum to 1 for sampling
        # Frechet mean is invariant under positive scaling of weights
        if weights is None:
            p_weights = np.array([1 / X.shape[0]] * X.shape[0])
        else:
            p_weights = weights / np.sum(weights)

        # set initial estimate
        mean_estimate, i = X[0], 0

        convergence = np.inf
        while convergence > self.epsilon and i < self.max_iter:
            # sample from datapoints
            sampled_point = np.random.choice(X, 1, p=p_weights)[0]

            # construct geodesic
            geodesic = self.space.metric.geodesic(mean_estimate, sampled_point)

            # new estimate is point 1/(i+2) of the way across the geodesic
            mean_estimate = geodesic(1 / (i + 2))
            i += 1

            # test for convergence lol not done yet
            convergence = 4

        if i == self.max_iter:
            logging.warning(
                f"Maximum number of iterations {self.max_iter} reached. The mean may be inaccurate."
            )

        self.estimate_ = mean_estimate
        return self

    def minimize(self, X, y=None, weights=None):
        r"""Empirical Frechet mean.

        Cheating but to ensure compatibility with gradient-descent-based methods.

        """
        return self.fit(X, y, weights)


class FrechetMean(BaseEstimator):
    r"""Empirical Frechet mean.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    method : str, {\'default\', \'adaptive\', \'batch\', \'sturms\'}
        Gradient descent method or iterative approach.
        The `adaptive` method uses a Levenberg-Marquardt style adaptation of
        the learning rate. The `batch` method is similar to the default
        method but for batches of equal length of samples. In this case,
        samples must be of shape [n_samples, n_batch, *space.shape].
        Optional, default: \'default\'.

    Attributes
    ----------
    estimate_ : array-like, shape=[*space.shape]
        If fit, Frechet mean.

    Notes
    -----
    * Required metric methods for general case:
        * `log`, `exp`, `squared_norm` (for convergence criteria)
    * Required metric methods for Sturm's algorithm:
        * `geodesic`
    """

    def __new__(cls, space, **kwargs):
        """Interface for instantiating proper algorithm."""
        if isinstance(space.metric, HypersphereMetric) and space.dim == 1:
            return CircleMean(space, **kwargs)

        elif _is_linear_metric(space.metric):
            return LinearMean(space, **kwargs)

        elif _is_elastic_metric(space.metric):
            return ElasticMean(space, **kwargs)

        elif _is_bhv_metric(space.metric):
            return SturmsMean(space, **kwargs)

        return super().__new__(cls)

    def __init__(self, space, method="default"):
        self.space = space

        self._method = None
        self.method = method

        self.estimate_ = None

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
            value, "method", ["default", "adaptive", "batch", "sturms"]
        )
        if value == self._method:
            return

        self._method = value
        MAP_OPTIMIZER = {
            "default": GradientDescent,
            "adaptive": AdaptiveGradientDescent,
            "batch": BatchGradientDescent,
            "sturms": SturmsMean,
        }
        self.optimizer = MAP_OPTIMIZER[value]()

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
        self.estimate_ = self.optimizer.minimize(
            space=self.space,
            points=X,
            weights=weights,
        )
        return self
