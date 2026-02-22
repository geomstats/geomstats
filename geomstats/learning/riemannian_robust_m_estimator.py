"""Riemannian Robust M-estimator Fitting.

Lead author: Jihyun Ryu.
"""

import abc
import inspect
import logging
import math
import time

from scipy.optimize import OptimizeResult
from sklearn.base import BaseEstimator

import geomstats.backend as gs
import geomstats.errors as error
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.geometric_median import GeometricMedian

numpy_backend = 'numpy' in gs.__name__


def _scalarmul(scalar, array):
    """Vectorized product."""
    return gs.einsum("n,n...->n...", scalar, array)


def _scalarmulsum(scalar, array):
    """Vectorized product sum."""
    return gs.einsum("n,n...->...", scalar, array)


def _gs_argsort(sorted_target):
    """Sort values and return sorted index from input.

    Notes
    -----
    The same as np.argsort method in numpy
    """
    sorted_idx = gs.array(
        [i for i, _ in sorted(enumerate(sorted_target), key=lambda x: x[1])]
        )
    return sorted_idx


def _rounding_array(array, decimal):
    """Round long floating number."""
    c = 10**decimal
    return gs.floor(array * c + 0.5) / c


def _set_midpoint(points):
    """Generate midpoint as initial point for Riemannian gradient descent.

    Parameters
    ----------
    points : array-like, shape=[n_samples, dim]
        Points.

    Returns
    -------
    Midpoint with respect to the first dimension axis values of the dataset.
    """
    n_points = points.shape[0]
    medpoint = int(n_points/2-1) if n_points % 2 == 0 else int((points.shape[0]-1)/2)
    first_coord = tuple([0]*(len(points.shape)-1))
    sorted_target = gs.array([points[i][first_coord] for i in range(n_points)])
    return points[_gs_argsort(sorted_target)[medpoint]]


def _set_mean_projection(space, points):
    """Generate mean-projection as initial point for Riemannian gradient descent.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    points : array-like, shape=[n_samples, dim]
        Points.

    Returns
    -------
    Projected value of Euclidean average point to given manifold.
    """
    mean_points = gs.mean(points, axis=0)
    mean_projection = space.projection(mean_points)
    return mean_projection


def riemannian_variance(
        space,
        points,
        base=None,
        weights=None,
        robust=False,
        get_centroid=False
):
    """Variance of (weighted) points wrt a base point.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    points : array-like, shape=[n_samples, *metric.shape]
        Points.
    base : array-like, shape=[*metric.shape]
        Estimated center of mass for variance.
    weights : array-like, shape=[n_samples,]
        Weights associated to the points.
        Optional, default: None.
    robust : boolean
        if True, use geometric median as the location parameter in the variance formula
        else, use Frechet mean as the location parameter in the variance formula.
        Optional, default: False.
    get_centroid : boolean
        if True, returns both variance and centroid.
        else, returns variance only.
        Optional, default: False.

    Returns
    -------
    var : float
        Weighted variance of the points.
    mean_estimate : array-like, shape=[dim]
        mean estimates of points provided.
    """
    if weights is None:
        n_points = gs.shape(points)[0]
        weights = gs.ones((n_points,))
    sum_weights = gs.sum(weights)

    if base is None:
        center = GeometricMedian(space, max_iter=1024) if robust else FrechetMean(space)
        center.fit(points)
        base = center.estimate_

    sq_dists = space.metric.squared_dist(base, points)
    var = weights * sq_dists

    var = gs.sum(var)
    var /= sum_weights

    mean_estimate = base
    if get_centroid:
        return var, mean_estimate
    return var


class BaseGradientDescent(abc.ABC):
    """Base class for gradient descent.

    Parameters
    ----------
    max_iter : int, optional
        Maximum number of iterations for the gradient descent.
    epsilon : float, optional
        Tolerance for stopping the gradient descent.
        default: 1e-6
    init_point : array-like, shape=[*metric.shape]
        Initial point.
        Optional, default : first. In this case the first sample
          of the input data is used.
    init_step_size : float
        Learning rate in the gradient descent.
        Optional, default: 1.
    autograd : bool,
        Perform by Autograd tools(valid when active geomstats backend is
          autograd or pytorch)
        Check gs.has_autodiff()
        Optional, default: False
    verbose : bool
        Level of verbosity to inform about convergence.
        Optional, default: False.
    perturbation_epsilon : float, optional
        Tiny movement parameter for the base when the base equals
          one of the value in data.
        In this case, the gradient is computed as NaN because denominator,
          the norm of log map, becomes 0.
        default: 1e-15
    """

    def __init__(
        self,
        max_iter=512,
        epsilon=1e-6,
        init_point=None,
        init_step_size=0.1,
        autograd=False,
        verbose=False,
        perturbation_epsilon=1e-15
    ):
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.init_step_size = init_step_size
        self.verbose = verbose
        self.init_point = init_point
        self.autograd = autograd
        self.perturbation_epsilon = perturbation_epsilon

    def fun__(self, fun, base):
        """Set given function as the function with one base point input.

        Parameters
        ----------
        function : method
            An M-estimator like method for gradient descent algorithm.
            The given method must must have variables (space, points, base)
              - space : Manifold
                  Equipped manifold.
              - points : array-like, shape=[n_samples, dim]
                  Points.
              - base : array-like, shape=[dim]
                  Points for the tangent space where gradient is computed.
            and optional variables (critical_value, weight)
              - critical_value : float or array-like,
                  parameter to manage the impact of outliers.
              - weights : array-like, shape=[n_samples,]
                  Weights associated to the points.
                  Optional, default: None.
        base : Not valid.
        """
        return lambda base: fun(self.points, base, self.weights)

    def _set_init_point(self, space, points, init_point_method):
        """Set initial starting point of algorithm."""
        if self.init_point is not None:
            current_mean = self.init_point
        else:
            if init_point_method == 'midpoint':
                current_mean = _set_midpoint(points)
            elif init_point_method == 'mean-projection':
                current_mean = _set_mean_projection(space, points)
            else:
                current_mean = points[0]
        return current_mean

    def _handle_jac(self, fun, point_ndim):
        """Define function for auto gradient."""
        if self.autograd:
            def fun_(x):
                """Set function output with loss and gradient with autodiff pkg."""
                value, grad = gs.autodiff.value_and_grad(fun, point_ndims=point_ndim)(x)
                return value, grad

        else:
            raise NotImplementedError("For now only working with autodiff.")

        return fun_

    def _handle_hess(self, fun, fun_hess):
        """Define Hessian for auto gradient(not used)."""
        if fun_hess is not None or (not self.autograd):
            fun_hess_ = fun_hess
            if callable(fun_hess):
                fun_hess_ = lambda x: fun_hess(gs.from_numpy(x))

            return fun_hess_

        return lambda x: gs.autodiff.hessian(fun)(gs.from_numpy(x))

    @abc.abstractmethod
    def minimize(
        self,
        space,
        points,
        critical_value,
        loss_grad_fun=False,
        weights=None,
        init_point_method=False
    ):
        """Perform gradient descent."""


class RiemannianAutoGradientDescent(BaseGradientDescent):
    """Riemannian Auto gradient descent.

    Notes
    -----
    1. Only works for the autograd/pytorch backend.
        (not working on default numpy backend)
    2. Not working for SPDLogEuclideanMetric on SPD Matrices manifolds.
    """

    def _perturbation_for_zero_distance(self, space, X, base):
        """Give perturbation if base has the exact same value as one given data."""
        equiv_w_base = gs.all(X == base, axis=1)
        if equiv_w_base.any():
            base = space.projection(base + self.perturbation_epsilon)
        return base

    def minimize(self, space, points, loss_grad_fun, weights=None, init_point_method='first'):
        """Perform auto grad descent by computing gradient of loss function.

        Parameters
        ----------
        space : Manifold
            Equipped manifold.
        points : array-like, shape=[n_samples, *metric.shape]
            Points to be averaged.
        loss_grad_fun : loss function
            Loss function given for computing gradient.
        weights : array-like, shape=[n_samples,],
            Weights associated to the samples.
            Optional, default: None, in which case it is equally weighted.
        init_point_method : str,
            first point initializing method. Optional,
            default : first. the first sample of the input data is used.
            mean-projection : projecting Euclidean mean onto the space given.
            midpoint : midpoint based on first dimension values

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            Result object with the following attributes:
            x : array-like, shape=[*metric.shape]
                Final estimate (the last iterate / mean estimate on the manifold).
            losses : list[float]
                Sequence of objective values recorded during optimization.
                Typically includes the initial value and values at each iteration.
            bases : list[array-like]
                Sequence of iterates (base points) produced by the algorithm.
                Each element has the same shape as `x`.
            n_iter : int
                Number of accepted iterations performed by the optimizer.
            time : float
                Wall-clock time in seconds elapsed during the optimization run.
        """
        lr = self.init_step_size

        current_loss = math.inf
        current_base = self._set_init_point(space, points, init_point_method)
        current_base = self._perturbation_for_zero_distance(
            space,
            points,
            current_base
        )
        current_iter = i = 0
        local_minima_gate = 0

        var = riemannian_variance(space, points, current_base, weights=weights)
        
        fun = '' if numpy_backend else self._handle_jac(loss_grad_fun, point_ndim=space.point_ndim)

        losses = [current_loss]
        bases = [current_base]
        tic = time.time()
        for i in range(self.max_iter):
            if numpy_backend:
                loss, grad = 0.01, space.to_tangent(points[1], current_base)
            else:
                loss, grad = fun(current_base)
                grad = space.to_tangent(grad, current_base)

            if gs.any(gs.isnan(grad)):
                logging.warning(
                    "NaN encountered in gradient at iter %d",
                    current_iter
                )
                lr /= 2
                local_minima_gate += 1
                if local_minima_gate >= 25:
                    logging.warning(
                        "NaN gradient value jumping at iteration %d...",
                        current_iter
                    )
                    lr = 10 * self.init_step_size
                    local_minima_gate = 0
                grad = current_base
            elif (loss >= current_loss) and (i > 0):
                lr /= 2
                local_minima_gate += 1
                if local_minima_gate >= 25:
                    logging.warning(
                        "local minima jumping at iteration %d...",
                        current_iter
                    )
                    lr = 10 * self.init_step_size
                    local_minima_gate = 0
            else:
                lr = self.init_step_size
                local_minima_gate = 0
                current_iter += 1

            if abs(space.metric.norm(grad, current_base)) < self.epsilon:
                if self.verbose:
                    logging.info(
                        "Tolerance threshold reached at iter %d",
                        current_iter
                    )
                break
            try:
                current_base = space.metric.exp(-lr * grad, current_base)
            except Exception as e:
                msg = str(e)
                if ('did not converge' in msg) and (isinstance(space, SPDMatrices)):
                    grad = grad + 1e-6*gs.eye(space.n)
                    current_base = space.metric.exp(-lr * grad, current_base)
                else:
                    raise
            if self.verbose and ((i+1) % 250 == 0):
                print(
                    f'{i+1}th iteration processing...  ' +
                    f'[{time.time()-tic:.2f} seconds]'
                )
                print(
                    f'base:{[_rounding_array(ee, 3) for ee in current_base]}, \
                    gradient:{[_rounding_array(ee, 3) for ee in grad]}, \
                    step size: {lr}, current loss: {_rounding_array(loss, 7)}]'
                )
            current_base = self._perturbation_for_zero_distance(
                space,
                points,
                current_base
            )
            current_loss = loss
            losses.append(current_loss)
            bases.append(current_base)
            var = riemannian_variance(space, points, current_base, weights=weights)
            if numpy_backend:
                raise NotImplementedError(
                    'Invalid to use on numpy backend. Try autograd, pytorch backend.'
                )

        if current_iter == self.max_iter:
            logging.warning(
                "Max number of iterations %d reached. The mean may be inaccurate",
                self.max_iter,
            )
        if self.verbose:
            logging.info(
                "Number of gradient evaluations: %d, "
                "Number of gradient iterations: %d, "
                " loss at termination: %e, "
                " standard deviation at termination: %e, ",
                i,
                current_iter,
                _rounding_array(current_loss, 6),
                _rounding_array(gs.sqrt(var), 6)
            )

        return OptimizeResult(
            x=current_base,
            losses=losses,
            bases=bases,
            n_iter=current_iter,
            time=time.time()-tic
        )


class GradientDescent(BaseGradientDescent):
    """Default gradient descent."""

    def minimize(
        self,
        space,
        points,
        loss_grad_fun=None,
        init_point_method='first'
    ):
        """Perform default gradient descent.

        Parameters
        ----------
        space : Manifold
            Equipped manifold.
        points : array-like, shape=[n_samples, *metric.shape]
                Points to be averaged.
        loss_grad_fun : loss function
            Loss function given for computing gradient.
        weights : array-like, shape=[n_samples,],
            Weights associated to the samples.
            Optional, default: None, in which case it is equally weighted.
        init_point_method : str,
            first point initializing method. Optional,
            default : first. In this case the first sample of the input data is used.
            mean-projection : projecting Euclidean mean onto the space given.
            midpoint : midpoint based on first dimension values

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            Result object with the following attributes:
            x : array-like, shape=[*metric.shape]
                Final estimate (the last iterate / mean estimate on the manifold).
            losses : list[float]
                Sequence of objective values recorded during optimization.
                Typically includes the initial value and values at each iteration.
            bases : list[array-like]
                Sequence of iterates (base points) produced by the algorithm.
                Each element has the same shape as `x`.
            n_iter : int
                Number of accepted iterations performed by the optimizer.
            time : float
                Wall-clock time in seconds elapsed during the optimization run.
        """
        n_points = gs.shape(points)[0]

        if n_points == 1:
            return n_points[0]

        mean = self._set_init_point(space, points, init_point_method)

        current_iter = 0

        tangent_norm_old = gs.sum(
            space.metric.norm(space.metric.log(points, mean), mean)
        )
        loss_v = tangent_norm_old
        step_size = self.init_step_size
        local_minima_gate = 0

        losses = []
        bases = []
        tic = time.time()
        while current_iter < self.max_iter:
            losses.append(loss_v)
            bases.append(mean)

            loss_v, gradient_value = loss_grad_fun(base=mean)
            gradient_value = space.to_tangent(gradient_value, mean)

            tangent_norm = gs.sum(space.metric.norm(gradient_value, mean))

            if self.verbose and (current_iter % 250 == 0):
                print(
                    f'{current_iter}th iteration processing...  '
                    f'[{time.time()-tic:.2f} seconds] '
                )
                print(
                    f'base:{[_rounding_array(ee, 3) for ee in mean]}, \
                    gradient:{[_rounding_array(ee, 3) for ee in gradient_value]}, \
                    step size: {step_size:.5f}, \
                    current loss(grad norm): {loss_v:.5f}(loss:{tangent_norm:.5f}]'
                )

            continuing_condition = gs.less_equal(self.epsilon * space.dim, tangent_norm)
            if not (continuing_condition or current_iter == 0):
                break

            estimate_next = space.metric.exp(-1 * step_size * gradient_value, mean)

            mean = estimate_next
            current_iter += 1

            if tangent_norm <= tangent_norm_old:
                tangent_norm_old = tangent_norm
                step_size = self.init_step_size
                local_minima_gate = 0
            elif tangent_norm > tangent_norm_old:
                step_size = max(0.001 * self.init_step_size, step_size / 2.0)
                local_minima_gate += 1
                if local_minima_gate >= 25:
                    logging.warning(
                        "local minima jumping at iteration %d...", current_iter
                    )
                    step_size = 10 * self.init_step_size
                    local_minima_gate = 0

        if current_iter == self.max_iter:
            logging.warning(
                "Maximum number of iterations %d reached. The mean may be inaccurate",
                self.max_iter
            )

        if self.verbose:
            var = gs.sum(space.metric.squared_dist(points, mean)) / (n_points - 1)
            logging.info(
                "n_iter: %d, final variance: %e, final loss: %e, gradient norm: %e",
                current_iter,
                var,
                loss_v,
                tangent_norm,
            )

        return OptimizeResult(
            x=mean,
            losses=losses,
            bases=bases,
            n_iter=current_iter,
            time=time.time()-tic
            )


class AdaptiveGradientDescent(BaseGradientDescent):
    """Adaptive gradient descent."""

    def minimize(
        self,
        space,
        points,
        loss_grad_fun=None,
        init_point_method='first'
    ):
        """Perform adaptive gradient descent.

        M-estimator mean of (weighted) points using adaptive time-steps
        The loss function optimized is given by M-estimator loss function.

        Adaptivity is done in a Levenberg-Marquardt style weighting variable tau
        between the first order and the second order Gauss-Newton gradient descent.

        Parameters
        ----------
        space : Manifold
            Equipped manifold.
        points : array-like, shape=[n_samples, *metric.shape]
            Points to be averaged.
        loss_grad_fun : loss function
            Loss function given for computing gradient.
        weights : array-like, shape=[n_samples,],
            Weights associated to the samples.
            Optional, default: None, in which case it is equally weighted.
        init_point_method : str,
            first point initializing method. Optional,
            default : first. In this case the first sample of the input data is used.
            mean-projection : projecting Euclidean mean onto the space given.
            midpoint : midpoint based on first dimension values

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            Result object with the following attributes:
            x : array-like, shape=[*metric.shape]
                Final estimate (the last iterate / mean estimate on the manifold).
            losses : list[float]
                Sequence of objective values recorded during optimization.
                Typically includes the initial value and values at each iteration.
            bases : list[array-like]
                Sequence of iterates (base points) produced by the algorithm.
                Each element has the same shape as `x`.
            n_iter : int
                Number of accepted iterations performed by the optimizer.
            time : float
                Wall-clock time in seconds elapsed during the optimization run.
        """
        n_points = gs.shape(points)[0]

        tau_max = 1e6 * self.init_step_size if self.init_step_size>1e3 else 1e6
        tau_mul_up = 1.6511111
        tau_min = 1e-6 * self.init_step_size if self.init_step_size>1e3 else 1e-6
        tau_mul_down = 0.1

        if n_points == 1:
            return points[0]

        current_mean = self._set_init_point(space, points, init_point_method)
        var = 0

        tau = self.init_step_size
        current_iter = 0
        stop_signal = False
        
        current_loss_v, current_gradient_value = loss_grad_fun(base=current_mean)
        current_gradient_value = space.to_tangent(current_gradient_value, current_mean)
        sq_norm_current_gradient_value = space.metric.squared_norm(
            current_gradient_value,
            current_mean
        )

        losses = []
        bases = []
        tic = time.time()
        while (
            (not stop_signal) and (current_iter < self.max_iter)
        ):
            current_iter += 1
            losses.append(current_loss_v)
            bases.append(current_mean)

            shooting_vector = -1 * tau * current_gradient_value
            next_mean = space.metric.exp(
                tangent_vec=shooting_vector, base_point=current_mean
            )

            next_loss_v, next_gradient_value = loss_grad_fun(base=next_mean)

            sq_norm_next_gradient_value = space.metric.squared_norm(
                next_gradient_value,
                next_mean
            )

            if next_loss_v <= current_loss_v:
                if current_loss_v - next_loss_v <= self.epsilon:
                    stop_signal = True
                    losses.append(next_loss_v)
                    bases.append(next_mean)
                    break
                else:
                    current_mean = next_mean
                    current_gradient_value = next_gradient_value
                    current_loss_v = next_loss_v
                    sq_norm_current_gradient_value = sq_norm_next_gradient_value
                    tau = min(tau_max, tau_mul_up * tau)
            elif abs(next_loss_v - current_loss_v) < 10*self.epsilon:
                tau = max(tau_min, tau_mul_down * tau)
            else:
                tau = max(tau_min, tau_mul_down * tau)

            if self.verbose and (current_iter % 250 == 0):
                print(
                    f'{current_iter}th iteration processing...  ' +
                    f'[{time.time()-tic:.2f} seconds]'
                )
                print(
                    f'base:{[_rounding_array(ee, 3) for ee in current_mean]}, ' +
                    f'gradient:{[_rounding_array(ee, 3) for ee in current_gradient_value]}, ' +
                    f'step size: {tau:.5f}, ' +
                    f'current loss(grad norm): {sq_norm_current_gradient_value:.2f}' +
                    f'(loss:{current_loss_v:.5f}]'
                )

            var = gs.sum(space.metric.squared_dist(points, current_mean))/(n_points-1)

        if current_iter == self.max_iter:
            logging.warning(
                "Maximum number of iterations %d reached. The mean may be inaccurate",
                self.max_iter,
            )

        if self.verbose:
            logging.info(
                "n_iter: %d, final variance: %e, final loss: %e, final_step_size: %e",
                current_iter,
                var,
                current_loss_v,
                tau,
            )

        return OptimizeResult(
            x=current_mean,
            losses=losses,
            bases=bases,
            n_iter=current_iter,
            time=time.time()-tic
            )


class LossBase:
    """Declare general parent class of losses."""

    def bind(self, space, points, weights=None, critical_value=None, autograd=False):
        """Generate empty bind method in parent class."""
        raise NotImplementedError
    def __call__(self, base, return_grad=False):
        """Generate empty call in parent class."""
        raise NotImplementedError


class BaseRiemannianLoss(LossBase):
    """Common init/bind/check for all losses (stateful bind style)."""

    def __init__(self, critical_value=1.345):
        self.critical_value = critical_value
        self._is_bound = False
        self._autograd = False

    def bind(self, space, points, critical_value=None, weights=None, autograd=False):
        """Bind fit context once per fit.

        Parameters
        ----------
        space : Manifold
            Equipped manifold.
        points : array-like, shape=[n_samples, *metric.shape]
            Training input samples.
        critical_value : float, or array-like, shape=[3,]
            Cutoff in the loss function to control outliers.
            If Hampel loss is applied, array-like shape available.
        weights : array-like, shape=[n_samples,]
            Weights associated to the samples.
            Optional, default: None, in which case it is equally weighted.
        autograd : bool
            If True, loss function only returns loss, and gradient of loss
              computed automatically by autograd process.
            else, loss function returns both loss and gradient of loss,
              for processing Riemannian gradient descent explicitly.

        Returns
        -------
        self : object
            Returns self.
        """
        self.space = space
        self.points = points

        n = gs.shape(points)[0]
        self.weights = gs.ones((n,)) if weights is None else weights
        self.sum_weights = gs.sum(self.weights)

        if critical_value is not None:
            self.critical_value = critical_value

        self._autograd = autograd
        self._is_bound = True
        return self

    def _check_bound(self):
        if not self._is_bound:
            raise RuntimeError("Loss is not bound: call bind(space, points, weights) first.")

    def __call__(self, base, return_grad=False):
        """Check bind, autograd policy, then delegates to subclass(base call).
        
        Parameters
        ----------
        base : array-like, shape=[*metric.shape]
            base point on manifold to get tangent space,
            and it will be the origin on the corresponding tangent space.
        return_grad: bool
            True when computing both loss and gradient of loss.
            False only for computing loss.

        Returns
        -------
        loss : float
            Weighted loss from the points provided.
        gradient : array-like, shape=[*metric.shape]
            gradient of loss from the points provided.
            This should be on the tangent space of the manifold(space) provided.
        """
        self._check_bound()
        if self._autograd:
            return_grad = False
        return self._call_impl(base, return_grad=return_grad)

    def _call_impl(self, base, return_grad=False):
        """Implement required for subclasses."""
        raise NotImplementedError


class RiemannianHuberLoss(BaseRiemannianLoss):
    """Huber loss.

    Loss:
        ρ(ξ) = ξ²              if ξ ≤ c
               2c(ξ - c/2)     if ξ > c
    Score:
        ψ(ξ) = 2ξ              if ξ ≤ c
               2c              if ξ > c
    Gradient:
        grad ρ = - 2 · (ψ(ξ)/ξ) · Log_x(p)

    Reference
    ---------
    [LJ2024] Lee, Jongmin, Jung, Sungkyu. “Huber means on Riemannian manifolds”,
    arXiv preprint arXiv:2407.15764, 2024. https://doi.org/10.48550/arXiv.2407.15764
    """

    def _call_impl(self, base, return_grad=False):
        c = self.critical_value
        logs = self.space.metric.log(point=self.points, base_point=base)
        distances = self.space.metric.norm(logs, base_point=base)

        close_distance = gs.less_equal(distances, c)
        loss = close_distance*(distances**2) + (~close_distance)*(2*c*(distances-c/2))
        loss = gs.sum(self.weights * loss) / self.sum_weights
        if not return_grad:
            return loss

        current_close_distance_gradient = _scalarmul(close_distance, logs)
        far_distance_huber_gradient = c * _scalarmul(1/(distances+10e-10), logs)
        current_far_distance_gradient = _scalarmul(
            (~close_distance),
            far_distance_huber_gradient
        )

        current_gradient_value = -1 * 2 * _scalarmulsum(
            self.weights,
            current_close_distance_gradient + current_far_distance_gradient
        ) / self.sum_weights
        return loss, self.space.to_tangent(current_gradient_value, base_point=base)


class RiemannianPseudoHuberLoss(BaseRiemannianLoss):
    """Pseudo Huber loss.

    Loss:
        ρ(ξ) = 2c²(sqrt(1 + (ξ/c)²) - 1)
    Score:
        ψ(ξ) = 2ξ / sqrt(1 + (ξ/c)²)
    Gradient:
        grad ρ = - (2 / sqrt(1 + (ξ/c)²)) · Log_x(p)

    Reference
    ---------
    [LJ2024] Lee, Jongmin, Jung, Sungkyu. “Huber means on Riemannian manifolds”,
    arXiv preprint arXiv:2407.15764, 2024. https://doi.org/10.48550/arXiv.2407.15764
    """

    def _call_impl(self, base, return_grad=False):
        c = self.critical_value
        logs = self.space.metric.log(point=self.points, base_point=base)
        distances = self.space.metric.norm(logs, base_point=base)

        squared_distances = distances ** 2
        loss = gs.sqrt(1 + squared_distances/(c**2)) - 1
        loss = 2*(c**2)*gs.sum(self.weights * loss) / self.sum_weights
        if not return_grad:
            return loss
        pseudo_huber_gradient = _scalarmul(1 / (gs.sqrt(1 + (distances/c)**2)), logs)
        if (c >= 0.01) & (c < 10):
            pseudo_huber_gradient_value = -1 * (
                2 * (c**2) * _scalarmulsum(self.weights, pseudo_huber_gradient)
            ) / self.sum_weights
        else:
            pseudo_huber_gradient_value = -1 * 2 * _scalarmulsum(
                self.weights,
                pseudo_huber_gradient
            ) / self.sum_weights

        return loss, self.space.to_tangent(pseudo_huber_gradient_value, base_point=base)


class RiemannianFairLoss(BaseRiemannianLoss):
    """Fair loss function.

    Loss:
        ρ(ξ) = c²(ξ/c - ln(1 + ξ/c))
    Score:
        ψ(ξ) = cξ / (c + ξ)
    Gradient:
        grad ρ = - (c / (c + ξ)) · Log_x(p)
    """

    def _call_impl(self, base, return_grad=False):
        c = self.critical_value
        logs = self.space.metric.log(point=self.points, base_point=base)
        distances = self.space.metric.norm(logs, base_point=base)

        loss = c**2 * (distances/c - gs.log(1 + distances/c))
        loss = gs.sum(self.weights * loss) / self.sum_weights
        if not return_grad:
            return loss

        grad = _scalarmul(c / (c + distances) , logs)
        grad = -1 * _scalarmulsum(self.weights, grad) / self.sum_weights
        return loss, self.space.to_tangent(grad, base_point=base)


class RiemannianCauchyLoss(BaseRiemannianLoss):
    """Cauchy loss function.

    Loss:
        ρ(ξ) = (c²/2) ln(1 + ξ²/c²)
    Score:
        ψ(ξ) = ξ / (1 + ξ²/c²)
    Gradient:
        grad ρ = - (1 / (1 + ξ²/c²)) · Log_x(p)
    """

    def _call_impl(self, base, return_grad=False):
        c = self.critical_value
        logs = self.space.metric.log(point=self.points, base_point=base)
        distances = self.space.metric.norm(logs, base_point=base)

        loss = c**2 / 2 * gs.log(1 + distances**2 / c**2)
        loss = gs.sum(self.weights * loss) / self.sum_weights
        if not return_grad:
            return loss
        grad = _scalarmul(c**2 / (c**2 + distances**2) , logs)
        grad = -1 * _scalarmulsum(self.weights, grad) / self.sum_weights
        return loss, self.space.to_tangent(grad, base_point=base)


class RiemannianBiweightLoss(BaseRiemannianLoss):
    """Tukey’s biweight loss function.

    Loss:
        ρ(ξ) = (c²/6)[1 - (1 - (ξ/c)²)³]    if ξ ≤ c
               c²/6                         if ξ > c
    Score:
        ψ(ξ) = ξ(1 - (ξ/c)² )²              if ξ ≤ c
               0                            if ξ > c
    Gradient:
        grad ρ = - (1 - (ξ/c)² )² · Log_x(p)  (for ξ ≤ c)
    """

    def _call_impl(self, base, return_grad=False):
        c = self.critical_value
        logs = self.space.metric.log(point=self.points, base_point=base)
        distances = self.space.metric.norm(logs, base_point=base)

        close_distance = gs.less_equal(distances, c)
        loss = (
            close_distance * (
                (c**2 / 6) * (1 - (1 - distances**2 / c**2)**3)
            ) + (~close_distance) * (c**2 / 6)
        )
        loss = gs.sum(self.weights * loss) / self.sum_weights
        if not return_grad:
            return loss
        grad = (
            _scalarmul(close_distance, _scalarmul((1 - distances**2/c**2)**2 , logs)) +
            _scalarmul((~close_distance), gs.zeros_like(logs))
        )
        grad = -1 * _scalarmulsum(self.weights, grad) / self.sum_weights
        return loss, self.space.to_tangent(grad, base_point=base)


class RiemannianWelschLoss(BaseRiemannianLoss):
    """Welsch loss function.

    Loss:
        ρ(ξ) = (c²/2)[1 - exp(-ξ²/c²)]
    Score:
        ψ(ξ) = ξ exp(-ξ²/c²)
    Gradient:
        grad ρ = - exp(-ξ²/c²) · Log_x(p)
    """

    def _call_impl(self, base, return_grad=False):
        c = self.critical_value
        logs = self.space.metric.log(point=self.points, base_point=base)
        distances = self.space.metric.norm(logs, base_point=base)

        loss = (0.5 * c**2 * (1 - gs.exp(- distances**2 / c**2)))
        loss = gs.sum(self.weights * loss) / self.sum_weights
        if not return_grad:
            return loss
        grad = _scalarmul(gs.exp(- distances**2 / c**2), logs)
        grad = -1 * _scalarmulsum(self.weights, grad) / self.sum_weights
        return loss, self.space.to_tangent(grad, base_point=base)


class RiemannianHampelLoss(BaseRiemannianLoss):
    """Hampel‐type redescending loss.

    Loss:
        ρ(ξ) = ξ²/2                                         if 0 ≤ ξ < a
               a(ξ - a/2)                                   if a ≤ ξ < b
               a(b - a/2) + a(c-b)/2 * [1 - ((c-ξ)/(c-b))²] if b ≤ ξ < c
               a(b - a/2) + a(c-b)/2                        if ξ ≥ c
    Score:
        ψ(ξ) = ξ                                        if 0 ≤ ξ < a
               a                                        if a ≤ ξ < b
               a(c-ξ)/(c-b)                             if b ≤ ξ < c
               0                                        if ξ ≥ c
    Gradient:
        grad ρ = - (ψ(ξ)/ξ) · Log_x(p)
    """

    def __init__(self, critical_value=(1.35, 2.7, 5.4)):
        """Hampel‐type redescending loss.

        Notes
        -----
        critical_value = (a_H, b_H, c_H)
            a_H: quadratic-to-linear change point
            b_H: linear-to-redescending change point
            c_H: redescending-to-constant change point
            If float type critical value is given, critical value is modified
            to (critical_value, 2*critical_value, 4*critical_value) automatically.
        """
        super().__init__(critical_value=critical_value)

    def _call_impl(self, base, return_grad=False):
        c = self.critical_value
        logs = self.space.metric.log(point=self.points, base_point=base)
        distances = self.space.metric.norm(logs, base_point=base)

        if isinstance(c, (float, int)):
            a = c * 1.0
            b = 2.0 * a
            c = 4.0 * a
        else:
            if len(c) != 3:
                raise ValueError(
                    "'critical_value' should be float/int or length-3 array-like value"
                    )
            a, b, c = c

        is_quadratic_region = gs.less_equal(distances, a)
        is_linear_region = (~is_quadratic_region) & gs.less_equal(distances, b)
        is_smooth_redescending = (
            (~gs.less_equal(distances, b)) & gs.less_equal(distances, c)
        )
        is_constant_region = ~gs.less_equal(distances, c)
        loss = (
            is_quadratic_region*(0.5 * distances**2) +
            is_linear_region*(a*distances - (a**2)/2) +
            is_smooth_redescending*(
                a*b - (a**2)/2 + a*(c-b)/2 * (1 - ((c-distances)/(c-b))**2)
            ) +
            is_constant_region*(a*b - (a**2)/2 + a*(c-b)/2)
        )
        loss = gs.sum(self.weights * loss) / self.sum_weights
        if not return_grad:
            return loss
        grad = (
            _scalarmul(is_quadratic_region, logs) +
            _scalarmul(is_linear_region, _scalarmul(a/(distances+1e-7), logs)) +
            _scalarmul(
                is_smooth_redescending,
                _scalarmul(a*(c-distances)/(c-b) * 1/(distances+1e-7), logs)
            ) +
            _scalarmul(is_constant_region, gs.zeros_like(logs))
        )
        grad = -1 * _scalarmulsum(self.weights, grad) / self.sum_weights
        return loss, self.space.to_tangent(grad, base_point=base)


class RiemannianCorrentropyLoss(BaseRiemannianLoss):
    """Correntropy loss function.

    Loss:
        ρ(ξ) = (1 / (c√(2π))) * [1 - exp(-ξ² / (2c²))]
    Score:
        ψ(ξ) = (ξ / (c³√(2π))) * exp(-ξ² / (2c²))
    Gradient:
        grad ρ = - (1 / (c³√(2π))) * exp(-ξ² / (2c²)) · Log_x(p)
    """

    def _call_impl(self, base, return_grad=False):
        c = self.critical_value
        logs = self.space.metric.log(point=self.points, base_point=base)
        distances = self.space.metric.norm(logs, base_point=base)

        norm_const = 1/(c * gs.sqrt(2 * gs.pi))
        loss = norm_const * (1 - gs.exp(- distances**2 / (2 * c**2)))
        loss = gs.sum(self.weights * loss) / self.sum_weights
        if not return_grad:
            return loss
        grad_const = 1/(c**3 * gs.sqrt(2 * gs.pi))
        grad = _scalarmul(grad_const * gs.exp(- distances**2 / (2 * c**2)), logs)
        grad = -1 * _scalarmulsum(self.weights, grad) / self.sum_weights
        return loss, self.space.to_tangent(grad, base_point=base)


class RiemannianLogisticLoss(BaseRiemannianLoss):
    """Logistic loss.

    Loss:
        ρ(ξ) = c² ln(cosh(ξ/c))
    Score:
        ψ(ξ) = c tanh(ξ/c)
    Gradient:
        grad ρ = - (c tanh(ξ/c) / ξ) · Log_x(p)
    """

    def _call_impl(self, base, return_grad=False):
        c = self.critical_value
        logs = self.space.metric.log(point=self.points, base_point=base)
        distances = self.space.metric.norm(logs, base_point=base)

        loss = (c**2 * gs.log(gs.cosh(distances / c)))
        loss = gs.sum(self.weights * loss) / self.sum_weights
        if not return_grad:
            return loss
        grad = _scalarmul((c * gs.tanh(distances / c) / (distances+1e-10)), logs)
        grad = -1 * _scalarmulsum(self.weights, grad) / self.sum_weights
        return loss, self.space.to_tangent(grad, base_point=base)


class RiemannianLorentzianLoss(BaseRiemannianLoss):
    """Lorentzian loss.

    Loss:
        ρ(ξ) = 1 - 1 / (1 + ξ²/(2c²))
    Score:
        ψ(ξ) = ξ / (c² (1 + ξ²/(2c²))²)
    Gradient:
        grad ρ = - (1 / (c² (1 + ξ²/(2c²))²)) · Log_x(p)
    """

    def _call_impl(self, base, return_grad=False):
        c = self.critical_value
        logs = self.space.metric.log(point=self.points, base_point=base)
        distances = self.space.metric.norm(logs, base_point=base)

        t = distances**2 / (2 * c**2)
        loss = 1 - 1/(1 + t)
        loss = gs.sum(self.weights * loss) / self.sum_weights
        if not return_grad:
            return loss
        grad = _scalarmul(1 / (c**2 * (1 + distances**2/(2*c**2))**2) , logs)
        grad = -1 * _scalarmulsum(self.weights, grad) / self.sum_weights
        return loss, self.space.to_tangent(grad, base_point=base)


class CallableLossAdapter(LossBase):
    """Adapting type of the given loss function method to LossBase class.
    
    Notes
    -----
    Required parameters of given function : space, points.
    It's possible not to have optional parameter names,
      e.g. weights/critical_value/return_grad(or loss_and_grad).
    
    """

    def __init__(self, fun):
        self.fun = fun
        self._is_bound = False

    def bind(self, space, points, weights=None, critical_value=None, autograd=False):
        """Bind fit context once per fit."""
        self.space = space
        self.points = points

        n = gs.shape(points)[0]
        self.weights = gs.ones((n,)) if weights is None else weights
        self.sum_weights = gs.sum(self.weights)

        if critical_value is not None:
            self.critical_value = critical_value

        self.autograd = autograd
        self._is_bound = True
        return self

    def __call__(self, base, return_grad=False):
        """Execute the custom loss function.

        Parameters
        ----------
        base : array-like, shape=[..., dim]
            The point on the manifold at which to evaluate the loss.
        return_grad : bool, default=False
            Whether to return the gradient. Ignored if `self.autograd` is True.

        Returns
        -------
        loss : float
            The computed Riemannian loss value.
        grad : array-like, optional
            The Riemannian gradient at `base`, returned only if `return_grad` is True.

        Raises
        ------
        RuntimeError
            If `bind` has not been called prior to execution.
        """
        if not self._is_bound:
            raise RuntimeError("Call bind(...) before using loss.")
        if self.autograd:
            return_grad = False

        return _call_with_supported_kwargs(
            self.fun,
            space=self.space,
            points=self.points,
            base=base,
            weights=self.weights,
            critical_value=self.critical_value,
            return_grad=return_grad,
            loss_and_grad=return_grad,
        )


def _safe_signature(fun):
    """Inspect parameters of function. covers callable, bound method, callable object."""
    try:
        return inspect.signature(fun)
    except (TypeError, ValueError):
        return None


def _call_with_supported_kwargs(fun, **kwargs):
    """Define function with only valid parameters."""
    sig = _safe_signature(fun)
    if sig is None:
        return TypeError("Valid parameters of the function are required")

    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fun(**kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in params}
    return fun(**filtered)


def squaredLoss(space,points,base,critical_value,weights=None,return_grad=True):
    """Compute simple L2 loss to give an example of valid custom function."""
    n = len(points)
    logs = space.metric.log(points,base)
    dists = space.metric.dist(points,base)

    loss = gs.sum(dists**2)/n
    if not return_grad:
        return loss
    grad = -2*gs.sum(logs,axis=0)/n
    return loss, space.to_tangent(grad,base)


class RiemannianRobustMEstimator(BaseEstimator):
    r"""Empirical Riemannian Robust Mean.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    critical_value : float, or array-like, shape=[3,]
        Cutoff in the loss function to control outliers.
        If Hampel loss is applied, array-like shape available.
        default: 0.
    method : str, {\'default\', \'adaptive\', \'autograd\' } - Batch not available now
        Gradient descent method.
        The `adaptive` method uses a Levenberg-Marquardt style adaptation of
        the learning rate. The `batch` method is similar to the default
        method but for batches of equal length of samples. In this case,
        samples must be of shape [n_samples, n_batch, *space.shape].
        The 'autograd' method uses autograd backend to calculate and
        optimize gradient automatically.
        - only avaliable with geomstats backend autograd, pytorch
        Optional, default: \'default\'.
    m_estimator : str, {'default','huber','pseudo-huber','cauchy','biweight','fair',
                'hampel','welsch','logistic','lorentzian','correntropy','custom'}
        M-estimator function, Huber loss for default

    Attributes
    ----------
    estimate_ : scipy.optimize.OptimizeResult
        Result object with the following attributes:
        x : array-like, shape=[*space.shape]
            Final estimate (the last iterate / mean estimate on the manifold).
        losses : list[float]
            Sequence of objective values recorded during optimization.
            Typically includes the initial value and values at each iteration.
        bases : list[array-like]
            Sequence of iterates (base points) produced by the algorithm.
            Each element has the same shape as `x`.
        n_iter : int
            Number of accepted iterations performed by the optimizer.
        time : float
            Wall-clock time in seconds elapsed during the optimization run.

    Notes
    -----
    * Required metric methods for general case:
        * `log`, `exp`, `squared_norm` (for convergence criteria)
    """

    def __new__(cls, space, critical_value=None, m_estimator='default', **kwargs):
        """Interface for instantiating proper algorithm."""
        return super().__new__(cls)

    def __init__(
            self,
            space,
            critical_value=None,
            m_estimator='default',
            init_point_method='first',
            method="default"
    ):
        """Set for initiate the estimator class."""
        if (numpy_backend) and (method == 'autograd'):
            raise NotImplementedError(
                "autograd method only available on autograd, pytorch backend. " +
                "use adaptive, default method on numpy backend."
            )

        self.space = space
        self.valid_m_estimators = [
            'default', 'huber', 'pseudo-huber', 'cauchy', 'biweight',
            'fair', 'hampel', 'welsch', 'logistic', 'lorentzian', 'correntropy',
            'custom'
        ]
        if m_estimator.lower() not in self.valid_m_estimators:
            raise ValueError(
                f"m_estimator must be in {', '.join(self.valid_m_estimators)}"
            )
        self.m_estimator = m_estimator.lower()
        self.critical_value = self._set_critical_value(critical_value)

        self.init_point_method = init_point_method

        self._method = None
        self.method = method
        self.is_autograd_method = self.method == 'autograd'

        self.estimate_ = None
        self.fun = None
        self.fun_provided = False

    def _set_critical_value(self, critical_value):
        """Set critical value for each m-estimator by 95% ARE for 1-d Euclidean space."""
        critical_value_for_95p_ARE = {
            'default': 1.345,
            'huber': 1.345,
            'pseudo-huber': 1.345,
            'cauchy': 2.3849,
            'biweight': 4.6851,
            'fair': 1.3998,
            'hampel': 1.35,
            'welsch': 2.9846,
            'logistic': 1.205,
            'lorentzian': 2.678,
            'correntropy': 2.1105}
        if (critical_value is None) & (self.m_estimator in critical_value_for_95p_ARE):
            return critical_value_for_95p_ARE[self.m_estimator]
        if (critical_value is None) & (self.m_estimator == 'custom'):
            critical_value = 1
        return critical_value

    def set(self, **kwargs):
        """Set optimizer parameters.

        Especially useful for one line instantiations.
        """
        for param_name, value in kwargs.items():
            if not hasattr(self.optimizer, param_name):
                raise ValueError(f"Unknown parameter {param_name}.")

            setattr(self.optimizer, param_name, value)
        return self

    def set_loss(self, fun=None):
        """Set loss function.

        Provide customized loss function to this instance by this method.
        or the pre-defined M-estimator function works for analysis.

        Parameters
        ----------
        fun : predefined function input,
            **must have the exact same argument names as below.**
          Customized functions should have input arguments
            (space, points, base):
            * space : manifold to learn algorithm
            * points : dataset for analysis
            * base : base point to get tangect space of the manifold(space)
          Customized functions are recommended(required on numpy backend)
            to have input arguments:
            * critical_value : to have robustness, we need to define critical point
                from which down-weight the impact of outliers
            * weights : if different weights are needed, required to have this argument

        Notes
        -----
        On autograd, pytorch backend, the output of the given function
            should be the loss computed in the function.
        To use this attribute on numpy backend, the outputs should be
            (loss, gradient) in the function given.
        ** For numpy backend, when custom loss function is given, be aware that 
            critical_value, weights, return_grad=True arguments must be given 
            although arguments are not used. **
        ex) def m_estimator_function(
                    space, points, base, critical_value, weights, return_grad=True
                    ):
                return loss, gradient
        """
        if fun is not None:
            if hasattr(fun, "bind") and callable(fun):
                self.fun = fun  # 이미 Loss object라고 가정
            else:
                self.fun = CallableLossAdapter(fun)
            self.fun_provided = True
        else:
            self.fun = self._set_m_estimator_loss()

        if (self.m_estimator != 'custom') and self.fun_provided:
            raise NotImplementedError(
                "Setting another M-estimator function is not valid. '" +
                "Try m_estimator='custom' input."
            )

    def _set_m_estimator_loss(self):
        """Return Loss_grad function based on given M-estimator name dynamically."""
        estimator_name = self.m_estimator.replace('-','_').lower()
        if estimator_name == 'default':
            estimator_name = 'huber'
        estimator_name = "".join([p.capitalize() for p in estimator_name.split("_")])
        class_name = f'Riemannian{estimator_name}Loss'

        try:
            m_estimator_loss_fun = globals()[class_name]
            return m_estimator_loss_fun(critical_value=self.critical_value)
        except (AttributeError, KeyError):
            valid_estimators = ', '.join(self.valid_m_estimators)
            raise ValueError(
                f"Not Supported M-estimator type : {self.m_estimator}, "
                f"must be in {{{valid_estimators}}}"
            )

    def _set_loss_function_gradientable(self, points, weights):
        """Set loss function utilizable before fitting.

        Pre-set the base point as the only parameter of the loss function 
            by making other variables fixed.
        If autograd not used and custom loss function given, 
            all parameters will be used in the gradient descent process.

        Parameters
        ----------
        points : array-like, shape=[n_samples, *metric.shape]
            Training input samples.
        weights : array-like, shape=[n_samples,]
            Weights associated to the samples.
            Optional, default: None, in which case it is equally weighted.

        Returns
        -------
        loss_with_base : callable
            Bound loss function that depends only on `base`.
            Signature: ``loss_with_base(base, return_grad=True)``.
            - If ``return_grad=False``: returns ``loss`` (float).
            - If ``return_grad=True``: returns ``(loss, grad)``, where ``grad`` is a
            tangent vector at ``base``.

        Notes
        -----
        This method binds the current fitting context (``space``, ``points``,
        ``weights``, and ``critical_value``) to ``self.fun`` via ``bind(...)``.
        Therefore, the returned callable only requires the base point as input.
        """
        if (self.m_estimator == 'custom') and (not self.fun_provided):
            raise NotImplementedError(
                'Custom M-estimator must be provided by set_loss() method.'
            )

        self.fun.bind(
            space=self.space,
            points=points,
            critical_value=self.critical_value,
            weights=weights,
            autograd=self.is_autograd_method
        )

        if self.is_autograd_method:
            def loss_with_base(base, return_grad=False):
                return self.fun(base, return_grad=return_grad)
            self.loss_with_base = loss_with_base
        else:
            def loss_with_base(base, return_grad=True):
                return self.fun(base, return_grad=return_grad)
            self.loss_with_base = loss_with_base
        return self.loss_with_base

    @property
    def method(self):
        """Gradient descent method."""
        return self._method

    @method.setter
    def method(self, value):
        """Gradient descent method."""
        error.check_parameter_accepted_values(
            value, "method", ["default", "adaptive", "autograd"]
        )
        if value == self._method:
            return

        self._method = value
        MAP_OPTIMIZER = {
            "default": GradientDescent,
            "adaptive": AdaptiveGradientDescent,
            "autograd": RiemannianAutoGradientDescent,
        }
        self.optimizer = MAP_OPTIMIZER[value]()
        if value in ['autograd']:
            self.set(autograd=True)

    def fit(self, X, y=None, weights=None):
        """Compute the empirical weighted Riemannian Robust M-estimator mean.

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
        if not self.fun_provided:
            self.set_loss()
        self._set_loss_function_gradientable(X, weights)

        if (not numpy_backend) and self.is_autograd_method:
            self.estimate_ = self.optimizer.minimize(
                space=self.space,
                points=X,
                loss_grad_fun=self.loss_with_base,
                weights=weights,
                init_point_method=self.init_point_method,
            )
            return self
        self.estimate_ = self.optimizer.minimize(
            space=self.space,
            points=X,
            loss_grad_fun=self.loss_with_base,
            init_point_method=self.init_point_method,
        )
        return self

    