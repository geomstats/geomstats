"""Riemannian Robust M-estimator Fitting

Lead author: Jihyun Ryu.
"""
"""
2nd upload of Riemannian robust M-estimators code
Fitting robust M-estimators on Riemannian Manifolds. By utilizing autograd and pytorch backends, 
we can also automatically calculate the gradient of loss function(either for M-estimators provided in the code(rho function) 
or loss function provided by user in a certain manner).

6/30 commit : Autograd, default base gradient, adaptive gradient code revision(iteration result tracking, escaping local minima, step size bounding)

"""

import abc
import logging
import math
import time

from sklearn.base import BaseEstimator
from scipy.optimize import OptimizeResult

import geomstats.backend as gs
import geomstats.errors as error
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.geometric_median import GeometricMedian
from geomstats.geometry.spd_matrices import SPDMatrices


def _scalarmul(scalar, array):
    return gs.einsum("n,n...->n...", scalar, array)


def _scalarmulsum(scalar, array):
    return gs.einsum("n,n...->...", scalar, array)


def _batchscalarmulsum(array_1, array_2):
    return gs.einsum("ni,ni...->i...", array_1, array_2)

def gs_argsort(sorted_target):
    sorted_idx = gs.array([i for i, _ in sorted(enumerate(sorted_target), key=lambda x: x[1])])
    return sorted_idx
    
def set_midpoint(points):
    n_points = points.shape[0]
    medpoint = int(n_points/2-1) if n_points%2==0 else int((points.shape[0]-1)/2)
    first_coord = tuple([0]*(len(points.shape)-1))
    sorted_target = gs.array([points[i][first_coord] for i in range(n_points)])
    return points[gs_argsort(sorted_target)[medpoint]]

def set_mean_projection(space,points):
    mean_points = gs.mean(points,axis=0)
    mean_projection = space.projection(mean_points)
    return mean_projection

def riemannian_variance(space, points, base=None, weights=None, robust=False, get_centroid=False):
    """Variance of (weighted) points wrt a base point.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    points : array-like, shape=[n_samples, dim]
        Points.
    base : array-like, shape=[dim]
        Estimates center of mass for variance
    weights : array-like, shape=[n_samples,]
        Weights associated to the points.
        Optional, default: None.
    robust : boolean
        if True, use geometric median as the location parameter in the variance formula
        else, use Frechet mean as the location parameter in the variance formula
    get_centroid : boolean
        if True, returns both variance and centroid. 
        else, returns variance only.

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
        if robust :
            GM = GeometricMedian(space,max_iter=1024)
            GM.fit(points)
            base = GM.estimate_
        else:
            FM = FrechetMean(space)
            FM.set(max_iter=1024)
            FM.fit(points)
            base = FM.estimate_
    
    sq_dists = space.metric.squared_dist(base, points)
    var = weights * sq_dists

    var = gs.sum(var)
    var /= sum_weights

    mean_estimate = base
    if get_centroid:
        return var,mean_estimate
    else:
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
        Optional, default : first. In this case the first sample of the input data is used.
    init_step_size : float
        Learning rate in the gradient descent.
        Optional, default: 1.
    autograd : bool, 
        Perform by Autograd tools(valid when active geomstats backend is autograd or pytorch)
        Check gs.has_autodiff()
        Optional, default: False
    verbose : bool
        Level of verbosity to inform about convergence.
        Optional, default: False.
    perturbation_epsilon : float, optional
        Tiny movement parameter for the base when the base equals one of the value in data.
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
        return lambda base: fun(self.points, base, self.weights)

    def _set_init_point(self, space, points, init_point_method):
        if self.init_point is not None:
            current_mean = self.init_point
        else:
            if init_point_method == 'midpoint':
                current_mean = set_midpoint(points)
            elif init_point_method == 'mean-projection':
                current_mean = set_mean_projection(space,points)
            else:
                current_mean = points[0]
        return current_mean

    def _handle_jac(self, fun, point_ndim):
        if self.autograd:
            def fun_(x):
                value, grad = gs.autodiff.value_and_grad(fun, point_ndims=point_ndim)(x)
                return value, grad

        else:
            raise NotImplementedError("For now only working with autodiff.")

        return fun_

    def _handle_hess(self, fun, fun_hess):
        if fun_hess is not None or (not self.autograd):
            fun_hess_ = fun_hess
            if callable(fun_hess):
                fun_hess_ = lambda x: fun_hess(gs.from_numpy(x))

            return fun_hess_

        return lambda x: gs.autodiff.hessian(fun)(gs.from_numpy(x))

    def _get_vector_transport(self, space):
        if hasattr(space.metric, "parallel_transport"):

            def vector_transport(tan_a, tan_b, base_point, _):
                return space.metric.parallel_transport(tan_a, base_point, tan_b)

        else:

            def vector_transport(tan_a, _, __, point):
                return space.to_tangent(tan_a, point)

        return vector_transport
       
    @abc.abstractmethod
    def minimize(self, space, points, critical_value, loss_grad_fun=False, weights=None, init_point_method=False):
        """Perform gradient descent."""
        pass


       
class RiemannianAutoGradientDescent(BaseGradientDescent):
    """Riemannian Auto gradient descent.
    
    Note
    ----
    1. Only works for the autograd/pytorch backend.
        (not working on default numpy backend)
    2. Not working for SPDLogEuclideanMetric on SPD Matrices manifolds.

    """

    def _perturbation_for_zero_distance(self, space, X, base):
        equiv_w_base = gs.all(X==base, axis=1)
        if equiv_w_base.any():
            base = space.projection(base + self.perturbation_epsilon)
        return base
   
    def minimize(self, space, points, fun, weights=None, init_point_method='first'):
        """Perform gradient descent by automatically computing gradient of loss function.
         
        Parameters
        ----------
        space : geomstats geometry class, manifold which the given data points are on.
        points : array-like, shape=[n_samples, *metric.shape]
                Points to be averaged.
        critical_value : the mininum threshold value for diminishing the effect of outlier.
        loss_grad_fun : loss function
        weights : array-like, shape=[n_samples,], optional
                explicitly weighting to the points - length must be the same as points.
        init_point_method : str,
            first point initializing method.
            Optional, default : first. In this case the first sample of the input data is used.
            mean-projection : averaging the data points and projection to the space(manifold) given.
            midpoint : sort by first dimension values and using the midpoint from the sorted order.

        """
        n_points = gs.shape(points)[0]
        lr = self.init_step_size
       
        current_loss = math.inf
        current_base = self._set_init_point(space, points, init_point_method)
        current_base = self._perturbation_for_zero_distance(space, points, current_base)
        current_iter = i = 0
        local_minima_gate = 0

        var = gs.sum(space.metric.squared_dist(points,current_base))/(n_points-1)
        
        fun = self._handle_jac(fun, point_ndim=space.point_ndim)        
       
        losses = [current_loss]
        bases = [current_base]
        tic = time.time()
        for i in range(self.max_iter):
            loss, grad = fun(current_base)
            grad = space.to_tangent(grad,current_base)
            
            if gs.any(gs.isnan(grad)):
                logging.warning(f"NaN encountered in gradient at iter {current_iter}")
                lr /= 2
                local_minima_gate += 1
                if local_minima_gate >= 25:
                    logging.warning(f"NaN gradient value jumping at iteration {current_iter}...")
                    lr = 100*self.init_step_size
                    local_minima_gate = 0
                grad = current_base
            elif loss >= current_loss and i > 0:
                lr /= 2
                local_minima_gate += 1
                if local_minima_gate >= 25:
                    logging.warning(f"local minima jumping at iteration {current_iter}...")
                    lr = 100*self.init_step_size
                    local_minima_gate = 0
            else:
                lr = self.init_step_size
                local_minima_gate = 0
                current_iter += 1

            if abs(space.metric.norm(grad,current_base)) < self.epsilon:
                if self.verbose:
                    logging.info(f"Tolerance threshold reached at iter {current_iter}")
                break
            try:
                current_base = space.metric.exp(-lr * grad, current_base)
            except Exception as e:
                msg = str(e)
                if ('did not converge' in msg) and (isinstance(space, SPDMatrices)):
                    grad = (grad + 1e-6*gs.eye(space.n))
                    current_base = space.metric.exp(-lr * grad, current_base)
                else:
                    raise
            if self.verbose and (i%50==0):
                print(f'{i}th iteration processing...  [{time.time()-tic:.2f} seconds]')
                print(f'base:{[ee.round(3) for ee in current_base]}, gradient:{[ee.round(3) for ee in grad]}, step size: {lr}, current loss: {round(loss,7)}]')
            current_base = self._perturbation_for_zero_distance(space, points, current_base)
            current_loss = loss
            losses.append(current_loss)
            bases.append(current_base)
            var = gs.sum(space.metric.squared_dist(points,current_base))/(n_points-1)
        
        if current_iter == self.max_iter:
            logging.warning(
                "Maximum number of iterations %d reached. The mean may be inaccurate",
                self.max_iter,
            )   
        if self.verbose:
            logging.info(
                f"Number of gradient evaluations: {i}, "
                f"Number of gradient iterations: {current_iter}, "
                f" loss at termination: {round(current_loss,6)}, "
                f" standard deviation at termination: {round(gs.sqrt(var),6)}, "
            )
         
        return OptimizeResult(x=current_base, losses=losses, bases=bases, n_iter=current_iter, time=time.time()-tic)


class GradientDescent(BaseGradientDescent):
    """Default gradient descent.
    
    """

    def minimize(self, space, points, critical_value, loss_grad_fun=None, weights=None, init_point_method='first'):
        """Perform default gradient descent.
        
        Parameters
        ----------
        space : geomstats geometry class, manifold which the given data points are on.
        points : array-like, shape=[n_samples, *metric.shape]
                Points to be averaged.
        critical_value : the mininum threshold value for diminishing the effect of outlier.
        loss_grad_fun : loss function
        weights : array-like, shape=[n_samples,], optional
                explicitly weighting to the points - length must be the same as points.
        init_point_method : str,
            first point initializing method.
            Optional, default : first. In this case the first sample of the input data is used.
            mean-projection : averaging the data points and projection to the space(manifold) given.
            midpoint : sort by first dimension values and using the midpoint from the sorted order.
            
        """
        
        n_points = gs.shape(points)[0]
        if weights is None:
            weights = gs.ones((n_points,))

        if n_points == 1:
            return n_points[0]
        
        mean = self._set_init_point(space, points, init_point_method)

        current_iter = 0
        sq_dist = 0.0
        var = 0.0

        tangent_norm_old = gs.sum(space.metric.norm(space.metric.log(points,mean), mean))
        loss_v = tangent_norm_old
        step_size = self.init_step_size
        local_minima_gate = 0

        losses = []
        bases = []
        tic = time.time()
        while current_iter < self.max_iter:
            losses.append(loss_v)
            bases.append(mean)

            loss_v, gradient_value = loss_grad_fun(points, mean, critical_value, weights, loss_and_grad=True)
            gradient_value = space.to_tangent(gradient_value, mean)

            tangent_norm = gs.sum(space.metric.norm(gradient_value, mean))

            if self.verbose and (current_iter%50==0):
                print(f'{current_iter}th iteration processing...  [{time.time()-tic:.2f} seconds] ')
                print(f'base:{[ee.round(3) for ee in mean]}, gradient:{[ee.round(3) for ee in gradient_value]}, step size: {step_size:.5f}, current loss(grad norm): {tangent_norm:.2f}(loss:{loss_v:.5f}]')

            continuing_condition = gs.less_equal(self.epsilon * space.dim, tangent_norm)
            if not (continuing_condition or current_iter == 0):
                break

            estimate_next = space.metric.exp(step_size * gradient_value, mean)

            var = gs.sum(space.metric.squared_dist(points,mean))/(n_points-1)
            sq_dist = space.metric.squared_norm(estimate_next, mean)

            mean = estimate_next
            current_iter += 1          

            if tangent_norm < tangent_norm_old:
                tangent_norm_old = tangent_norm
                step_size = self.init_step_size
                local_minima_gate = 0
            elif tangent_norm > tangent_norm_old:
                step_size = max(0.001, step_size / 2.0)
                local_minima_gate += 1
                if local_minima_gate >= 25:
                    logging.warning(f"local minima jumping at iteration {current_iter}...")
                    step_size = 100 * self.init_step_size
                    local_minima_gate = 0
            
            

        if current_iter == self.max_iter:
            logging.warning(
                "Maximum number of iterations {} reached. The mean may be inaccurate".format(
                self.max_iter
                )
            )

        if self.verbose:
            logging.info(
                "n_iter: {}, final variance: {:.5f}, final dist: {:.5f}, gradient norm: {:.5f}".format(
                    current_iter, var, sq_dist, tangent_norm
                )
            )

        return OptimizeResult(x=mean, losses=losses, bases=bases, n_iter=current_iter, time=time.time()-tic)


class AdaptiveGradientDescent(BaseGradientDescent):
    """Adaptive gradient descent."""

    def minimize(self, space, points, critical_value, loss_grad_fun=None, weights=None, init_point_method='first'):
        """Perform adaptive gradient descent.
        M-estimator mean of (weighted) points using adaptive time-steps
        The loss function optimized is given by M-estimator loss function.
        
        Adaptivity is done in a Levenberg-Marquardt style weighting variable tau
        between the first order and the second order Gauss-Newton gradient descent.

        Parameters
        ----------
        space : geomstats geometry class, manifold which the given data points are on.
        points : array-like, shape=[n_samples, *metric.shape]
            Points to be averaged.
        critical_value : the mininum threshold value for diminishing the effect of outlier.
        loss_grad_fun : loss function
        weights : array-like, shape=[n_samples,], optional
            explicitly weighting to the points - length must be the same as points.
        init_point_method : str,
            first point initializing method.
            Optional, default : first. In this case the first sample of the input data is used.
            mean-projection : averaging the data points and projection to the space(manifold) given.
            midpoint : sort by first dimension values and using the midpoint from the sorted order.
        """
        n_points = gs.shape(points)[0]

        tau_max = 1e6
        tau_mul_up = 1.6511111
        tau_min = 1e-6
        tau_mul_down = 0.1

        if n_points == 1:
            return points[0]
       
        current_mean = self._set_init_point(space, points, init_point_method)

        if weights is None:
            weights = gs.ones((n_points,))
        #sum_weights = gs.sum(weights)

        tau = self.init_step_size
        current_iter = 0

        loss_v, current_gradient_value = loss_grad_fun(points, current_mean, critical_value, weights, loss_and_grad=True)
        current_gradient_value = space.to_tangent(current_gradient_value, current_mean)
        sq_norm_current_gradient_value = space.metric.squared_norm(current_gradient_value, current_mean)

        losses = []
        bases = []
        tic = time.time()
        while (
            sq_norm_current_gradient_value > self.epsilon**2 and current_iter < self.max_iter
        ):
            current_iter += 1
            losses.append(gs.sqrt(sq_norm_current_gradient_value))
            bases.append(current_mean)

            shooting_vector = tau * current_gradient_value
            next_mean = space.metric.exp(
                tangent_vec=shooting_vector, base_point=current_mean
            )
           
            loss_v, next_gradient_value = loss_grad_fun(points, next_mean, critical_value, weights, loss_and_grad=True)
                       
            sq_norm_next_gradient_value = space.metric.squared_norm(next_gradient_value, next_mean)

            if sq_norm_next_gradient_value < sq_norm_current_gradient_value:
                current_mean = next_mean
                current_gradient_value = next_gradient_value
                sq_norm_current_gradient_value = sq_norm_next_gradient_value
                tau = min(tau_max, tau_mul_up * tau)
            else:
                tau = max(tau_min, tau_mul_down * tau)
            
            if self.verbose and (current_iter%50==0):
                print(f'{current_iter}th iteration processing...  [{time.time()-tic:.2f} seconds]')
                print(f'base:{[ee.round(3) for ee in current_mean]}, gradient:{[ee.round(3) for ee in current_gradient_value]}, step size: {tau:.5f}, current loss(grad norm): {sq_norm_current_gradient_value:.2f}(loss:{loss_v:.5f}]')

            var = gs.sum(space.metric.squared_dist(points,current_mean))/(n_points-1)

        if current_iter == self.max_iter:
            logging.warning(
                "Maximum number of iterations %d reached. The mean may be inaccurate",
                self.max_iter,
            )

        if self.verbose:
            logging.info(
                "n_iter: %d, final variance: %e, final dist: %e, final_step_size: %e",
                current_iter,
                var,
                sq_norm_current_gradient_value,
                tau,
            )

        return OptimizeResult(x=current_mean, losses=losses, bases=bases, n_iter=current_iter, time=time.time()-tic)



class RiemannianRobustMestimator(BaseEstimator):
    r"""Empirical Riemannian Robust Mean.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    critical_value : critical value of Robust M-estimators.
        if critical_value is zero, find the critical value satisfying 95% ARE
        compared to normal distribution with scale parameter 1
        according to M-estimator function input respectively.
        default : 0
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
    m_estimator : str, {'default','huber','pseudo_huber','cauchy','biweight',
                        'fair','hampel','welsch','logistic','lorentzian','correntropy'}
        m_estimator function
        huber loss for default

    Attributes
    ----------
    estimate_ : OptimizeResult instance, 
        x : If fit, Huber mean. or mean by pre-specified loss function.
        losses : containing all loss results throughout learning
        bases : containing all base results throughout learning
        n_iter : total iteration counts
        time : elapsed time to finish learning
    Notes
    -----
    * Required metric methods for general case:
        * `log`, `exp`, `squared_norm` (for convergence criteria)
    """

    def __new__(cls, space, critical_value, m_estimator, **kwargs):
        """Interface for instantiating proper algorithm."""
        # if critical_value > 10e7:
        #     return FrechetMean(space, **kwargs)
        # elif critical_value < 10e-7:
        #     return GeometricMedian(space, **kwargs)

        return super().__new__(cls)

    def __init__(self, space, critical_value = 0, m_estimator='default', init_point_method='first', method="default"):
        self.space = space
        self.valid_m_estimators = ['default','huber','pseudo_huber','cauchy','biweight','fair','hampel','welsch','logistic','lorentzian','correntropy']
        assert m_estimator.lower() in self.valid_m_estimators,\
                f"m_estimator must be in {','.join(self.valid_m_estimators)}"
        self.m_estimator = m_estimator.lower()
        self.critical_value = self._set_critical_value(critical_value)
       
        assert init_point_method in ['first','midpoint','mean-projection']
        self.init_point_method = init_point_method
       
        self._method = None
        self.method = method
       
        self.estimate_ = None
        self.fun = None
        self.fun_provided = False
    
    def _set_critical_value(self,critical_value):
        critical_value_for_95p_ARE = {
            'default':1.345,
            'huber':1.345,
            'pseudo_huber':1.345,
            'cauchy':2.3849,
            'biweight':4.6851,
            'fair':1.3998,
            'hampel':1.35,
            'welsch':2.9846,
            'logistic':1.205,
            'lorentzian':2.678,
            'correntropy':2.1105}
        if critical_value == 0:
            return critical_value_for_95p_ARE[self.m_estimator]
        else:
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

    def _set_loss(self,fun=None):
        """Set loss function.

        Provide customized loss function to this instance by this method.
        or the pre-defined M-estimator function works for analysis.

        Parameters
        ----------
        fun : predefined function argument, * must have the same argument names as below.
          Customized functions should have input arguments (space, points, base):
            * space : manifold to learn algorithm
            * points : dataset for analysis
            * base : base point to get tangect space of the manifold(space)
          Customized functions are recommended to have input arguments:
            * critical_value : to have robustness, we need to define critical point
                from which down-weight the impact of outliers
            * weights : if different weighting is needed, required to have this argument
        """
        if fun is not None:
            self.fun = fun
            self.fun_provided = True
        if not self.fun_provided:
            self.fun = self._set_m_estimator_loss()
        arguments = self.fun.__code__.co_varnames[:self.fun.__code__.co_argcount]
       
        if not self.fun_provided:
            self.loss_with_base = lambda base: self.fun(
                    points=self.points, base=base, critical_value=self.critical_value)
        else:
            print(self.fun)
            if ('weights' in arguments) and ('critical_value' in arguments):
                self.loss_with_base = lambda base: self.fun(
                    space=self.space, points=self.points, base=base, critical_value=self.critical_value, weights=self.weights)
            elif ('weights' not in arguments) and ('critical_value' in arguments):
                self.loss_with_base = lambda base: self.fun(
                    space=self.space, points=self.points, base=base, critical_value=self.critical_value)
            elif ('weights' in arguments) and ('critical_value' not in arguments):
                self.loss_with_base = lambda base: self.fun(
                    space=self.space, points=self.points, base=base, weights=self.weights)
            else:
                self.loss_with_base = lambda base: self.fun(
                    space=self.space, points=self.points, base=base)
       
        return self.loss_with_base

    def _set_m_estimator_loss(self):
        if (self.m_estimator in ['default','huber']):
            return self._riemannian_huber_loss_grad
        elif (self.m_estimator == 'pseudo_huber'):
            return self._riemannian_pseudo_huber_loss_grad
        elif self.m_estimator == 'cauchy':
            return self._riemannian_cauchy_loss_grad
        elif self.m_estimator == 'biweight':
            return self._riemannian_biweight_loss_grad
        elif self.m_estimator == 'fair':
            return self._riemannian_fair_loss_grad
        elif self.m_estimator == 'hampel':
            return self._riemannian_hampel_loss_grad
        elif self.m_estimator == 'welsch':
            return self._riemannian_welsch_loss_grad
        elif self.m_estimator == 'logistic':
            return self._riemannian_logistic_loss_grad
        elif self.m_estimator == 'lorentzian':
            return self._riemannian_lorentzian_loss_grad
        elif self.m_estimator == 'correntropy':
            return self._riemannian_correntropy_loss_grad
        else:
            raise ValueError(f"Not Supported M-estimator type : {self.m_estimator}, must be in {','.join(self.valid_m_estimators)}")
   
    def _set_weights(self,n,weights=None):
        if weights is None:
            weights = gs.ones(n)
        else:
            weights = gs.asarray(weights)
        sum_weights = gs.sum(weights)
        return weights,sum_weights        
   
    def _riemannian_huber_loss_grad(self, points, base, critical_value=1.345, weights=None, loss_and_grad=False):
        """Huber loss
        ρ(ξ) = ξ²          if ξ ≤ c
               2c(ξ-c/2)   if ξ > c

        Parameters
        ----------
        points : array_like of shape [n, ...], data points.
        base: base point on manifold to get tangent space,
            and base point will be origin on the related tangent space.
        critical_value : float or array_like of shape [3,]
            cutoff in the loss function to control outliers
        weights : array_like of shape [n,], default None (equal weights)
            optional, default: None
        loss_and_grad: bool
            True only for computing loss.
            False when computing both loss and gradient of loss.
        
        Returns
        -------
        loss : float
            Weighted loss from the points provided.
        gradient : array-like, shape=[dim,]
            gradient of loss from the points provided.
            This should be on the tangent space of the manifold(space) provided.

        Reference
        ---------
        [LJ2024] Lee, Jongmin, Jung, Sungkyu. “Huber means on Riemannian manifolds”, arXiv preprint arXiv:2407.15764, 2024. https://doi.org/10.48550/arXiv.2407.15764
        """
        n = points.shape[0]
        weights, sum_weights = self._set_weights(n,weights=None)
        self.critical_value = critical_value
        c = self.critical_value
       
        logs = self.space.metric.log(point=points, base_point=base)
        distances = self.space.metric.norm(logs, base)
        
        close_distance = gs.less_equal(distances,c)
        loss = close_distance*(distances**2) + (1-close_distance)*(2*c*(distances-c/2))
        loss = gs.sum( weights * loss ) / sum_weights
        if not loss_and_grad:
            return loss
        
        current_close_distance_gradient = _scalarmul(close_distance, logs)
        far_distance_huber_gradient = c * _scalarmul(1/(distances+10e-10), logs)
        current_far_distance_gradient = _scalarmul((1-close_distance), far_distance_huber_gradient)
       
        current_gradient_value = 2 * _scalarmulsum(weights, current_close_distance_gradient + current_far_distance_gradient) \
                                        / sum_weights
        return loss, self.space.to_tangent(current_gradient_value,base_point=base)
   
    def _riemannian_pseudo_huber_loss_grad(self, points, base, critical_value=1.345, weights=None, loss_and_grad=False):
        """pseudo Huber loss

        ρ(ξ) = 2c²{sqrt(1+(ξ/c)²)-1}

        Parameters
        ----------
        points : array_like of shape [n, ...], data points.
        base: base point on manifold to get tangent space,
            and base point will be origin on the related tangent space.
        critical_value : float or array_like of shape [3,]
            cutoff in the loss function to control outliers
        weights : array_like of shape [n,], default None (equal weights)
            optional, default: None
        loss_and_grad: bool
            True only for computing loss.
            False when computing both loss and gradient of loss.
        
        Returns
        -------
        loss : float
            Weighted loss from the points provided.
        gradient : array-like, shape=[dim,]
            gradient of loss from the points provided.
            This should be on the tangent space of the manifold(space) provided.

        Reference
        ---------
        [LJ2024] Lee, Jongmin, Jung, Sungkyu. “Huber means on Riemannian manifolds”, arXiv preprint arXiv:2407.15764, 2024. https://doi.org/10.48550/arXiv.2407.15764
        """
        n = points.shape[0]
        weights, sum_weights = self._set_weights(n,weights=None)
        self.critical_value = critical_value
        c = self.critical_value
       
        logs = self.space.metric.log(point=points, base_point=base)
        squared_distances = self.space.metric.squared_norm(logs,base)
        distances = gs.sqrt(squared_distances)
        loss = gs.sqrt(1 + squared_distances/(c**2)) - 1
        loss = 2*(c**2)*gs.sum(weights * loss) / sum_weights
        if not loss_and_grad:
            return loss      
        pseudo_huber_gradient = _scalarmul(1 / (gs.sqrt(1 + (distances/c)**2)), logs)
        #pseudo_huber_gradient_value = 2 * (critical_value**2) * _scalarmulsum(weights, pseudo_huber_gradient) / sum_weights
        if (0.01 <= c) & (c < 10):
            pseudo_huber_gradient_value = 2 * (c**2) * _scalarmulsum(weights, pseudo_huber_gradient) / sum_weights
        else:
            pseudo_huber_gradient_value = 2 * _scalarmulsum(weights, pseudo_huber_gradient) / sum_weights
           
        return loss, self.space.to_tangent(pseudo_huber_gradient_value,base_point=base)

   
    def _riemannian_fair_loss_grad(self, points, base, critical_value=1.3998, weights=None, loss_and_grad=False):
        """Fair loss function
        
        ρ(ξ)=c²(ξ/c - ln(1+ξ/c))
        gradients: ρ'(ξ)=c·ξ/(c+ξ) → grad_i = -(c/(c+ξ))·v

        Parameters
        ----------
        points : array_like of shape [n, ...], data points.
        base: base point on manifold to get tangent space,
            and base point will be origin on the related tangent space.
        critical_value : float or array_like of shape [3,]
            cutoff in the loss function to control outliers
        weights : array_like of shape [n,], default None (equal weights)
            optional, default: None
        loss_and_grad: bool
            True only for computing loss.
            False when computing both loss and gradient of loss.
        
        Returns
        -------
        loss : float
            Weighted loss from the points provided.
        gradient : array-like, shape=[dim,]
            gradient of loss from the points provided.
            This should be on the tangent space of the manifold(space) provided.
        """
        
        n = len(points)
        weights,sum_weights = self._set_weights(n,weights=None)
        self.critical_value = critical_value
        c = self.critical_value
       
        logs  = self.space.metric.log(point=points, base_point=base)
        distances = self.space.metric.norm(logs, base_point=base)
        loss = c**2 * (distances/c - gs.log(1 + distances/c))
        loss = gs.sum(weights * loss) / sum_weights
        if not loss_and_grad:
            return loss
       
        grad = _scalarmul(c / (c+distances) , logs)
        grad = _scalarmulsum(weights, grad) / sum_weights
        return loss, self.space.to_tangent(grad,base_point=base)
   
    def _riemannian_cauchy_loss_grad(self, points, base, critical_value=2.3849, weights=None, loss_and_grad=False):
        """Cauchy loss function
        
        ρ(ξ) = (c²/2)·ln(1 + ξ²/c²)
        gradients: ρ'(ξ)=c²/(c²+ξ²) → grad_i = -(c²/(c²+ξ²))·v

        Parameters
        ----------
        points : array_like of shape [n, ...], data points.
        base: base point on manifold to get tangent space,
            and base point will be origin on the related tangent space.
        critical_value : float or array_like of shape [3,]
            cutoff in the loss function to control outliers
        weights : array_like of shape [n,], default None (equal weights)
            optional, default: None
        loss_and_grad: bool
            True only for computing loss.
            False when computing both loss and gradient of loss.
        
        Returns
        -------
        loss : float
            Weighted loss from the points provided.
        gradient : array-like, shape=[dim,]
            gradient of loss from the points provided.
            This should be on the tangent space of the manifold(space) provided.
        """
        
        n = len(points)
        weights,sum_weights = self._set_weights(n,weights=None)
        self.critical_value = critical_value
        c = self.critical_value
       
        logs  = self.space.metric.log(point=points, base_point=base)
        distances = self.space.metric.norm(logs, base_point=base)
        loss = c**2 / 2 * gs.log( 1 + distances**2 / c**2 )
        loss = gs.sum(weights * loss) / sum_weights
        if not loss_and_grad:
            return loss
        grad = _scalarmul(c**2 / (c**2 + distances**2) , logs)
        grad = _scalarmulsum(weights, grad) / sum_weights
        return loss, self.space.to_tangent(grad,base_point=base)
       
   
    def _riemannian_biweight_loss_grad(self, points, base, critical_value=4.6851, weights=None, loss_and_grad=False):
        """Tukey’s biweight loss function:
          ρ(ξ) = (c²/6)[1 - (1 - ξ²/c²)³],  |ξ|≤c
                 c²/6,                      |ξ|>c
          gradients: ρ'(ξ)=ξ·(1 - ξ²/c²)² → grad_i = -(1 - (ξ/c)²)² · v
        
        Parameters
        ----------
        points : array_like of shape [n, ...], data points.
        base: base point on manifold to get tangent space,
            and base point will be origin on the related tangent space.
        critical_value : float or array_like of shape [3,]
            cutoff in the loss function to control outliers
        weights : array_like of shape [n,], default None (equal weights)
            optional, default: None
        loss_and_grad: bool
            True only for computing loss.
            False when computing both loss and gradient of loss.
        
        Returns
        -------
        loss : float
            Weighted loss from the points provided.
        gradient : array-like, shape=[dim,]
            gradient of loss from the points provided.
            This should be on the tangent space of the manifold(space) provided.
        """
        n = len(points)
        weights,sum_weights = self._set_weights(n,weights=None)
        self.critical_value = critical_value
        c = self.critical_value
       
        logs  = self.space.metric.log(point=points, base_point=base)
        distances = self.space.metric.norm(logs, base_point=base)
        close_distance = gs.less_equal(distances,c)
        loss = close_distance * ((c**2 / 6) * (1 - (1 - distances**2 / c**2)**3)) + (1 - close_distance) * (c**2 / 6)
        loss = gs.sum(weights * loss) / sum_weights
        if not loss_and_grad:
            return loss
        grad =  (_scalarmul(close_distance, _scalarmul((1 - distances**2/c**2)**2 , logs )) + \
                        _scalarmul( (1 - close_distance) , gs.zeros_like(logs)) )
        grad = _scalarmulsum(weights, grad) / sum_weights
        return loss, self.space.to_tangent(grad,base_point=base)
   
    def _riemannian_welsch_loss_grad(self, points, base, critical_value=2.9846, weights=None, loss_and_grad=False):
        """Welsch loss function
          
        ρ(ξ) = (c²/2)[1 - exp(-ξ²/c²)]
        gradients: ρ'(ξ)=ξ·exp(-ξ²/c²) → grad_i = -exp(-ξ²/c²)·v

        Parameters
        ----------
        points : array_like of shape [n, ...], data points.
        base: base point on manifold to get tangent space,
            and base point will be origin on the related tangent space.
        critical_value : float or array_like of shape [3,]
            cutoff in the loss function to control outliers
        weights : array_like of shape [n,], default None (equal weights)
            optional, default: None
        loss_and_grad: bool
            True only for computing loss.
            False when computing both loss and gradient of loss.
        
        Returns
        -------
        loss : float
            Weighted loss from the points provided.
        gradient : array-like, shape=[dim,]
            gradient of loss from the points provided.
            This should be on the tangent space of the manifold(space) provided.
        """
        n = len(points)
        weights,sum_weights = self._set_weights(n,weights=None)
        self.critical_value = critical_value
        c = self.critical_value
   
        logs  = self.space.metric.log(point=points, base_point=base)
        distances = self.space.metric.norm(logs, base_point=base)
        loss = (0.5 * c**2 * (1 - gs.exp(- distances**2 / c**2)))
        loss = gs.sum(weights * loss) / sum_weights
        if not loss_and_grad:
            return loss
        grad = _scalarmul( gs.exp(- distances**2 / c**2), logs)
        grad = _scalarmulsum(weights, grad) / sum_weights
        return loss, self.space.to_tangent(grad,base_point=base)
   
   
    def _riemannian_hampel_loss_grad(self, points, base, critical_value=1.35, weights=None, loss_and_grad=False):
        """Hampel‐type redescending loss        
        
        grad = - (ρ'(ξ)/ξ) · v
        
        Parameters
        ----------
        points : array_like of shape [n, ...], data points.
        base: base point on manifold to get tangent space,
            and base point will be origin on the related tangent space.
        critical_value : float or array_like of shape [3,]
            cutoff in the loss function to control outliers
        weights : array_like of shape [n,], default None (equal weights)
            optional, default: None
        loss_and_grad: bool
            True only for computing loss.
            False when computing both loss and gradient of loss.
        
        Returns
        -------
        loss : float
            Weighted loss from the points provided.
        gradient : array-like, shape=[dim,]
            gradient of loss from the points provided.
            This should be on the tangent space of the manifold(space) provided.
        
        Notes
        -----
        critical_value = (a_H, b_H, c_H)
            - a_H: quadratic-to-linear change point
            - b_H: linear-to-redescending change point
            - c_H: redescending-to-constant change point
            If float type critical value is given, critical value is modified to 
                (critical_value, 2*critical_value, 4*critical_value)
        """
        
        self.critical_value = critical_value
        if isinstance(self.critical_value,float) or isinstance(self.critical_value,int):
            a = self.critical_value*1.0
            b = 2*a
            c = 4*a
        else:
            assert len(self.critical_value) == 3
            a, b, c = self.critical_value
        n = len(points)
        weights,sum_weights = self._set_weights(n,weights=None)
   
        logs  = self.space.metric.log(point=points, base_point=base)
        distances = self.space.metric.norm(logs, base_point=base)
   
        is_quadratic_region = distances<=a
        is_linear_region = (distances>a) & (distances<=b)
        is_smooth_redescending = (distances>b) & (distances<=c)
        is_constant_region = (distances>c)
        loss = is_quadratic_region*(0.5 * distances**2) + is_linear_region*(a*distances - (a**2)/2) +\
                    is_smooth_redescending*(a*b - (a**2)/2 + a*(c-b)/2 * (1 - ((c-distances)/(c-b))**2)) +\
                    is_constant_region*(a*b - (a**2)/2 + a*(c-b)/2)
        loss = gs.sum(weights * loss) / sum_weights
        if not loss_and_grad:
            return loss
        grad = ( _scalarmul(is_quadratic_region,logs) + _scalarmul(is_linear_region, _scalarmul(a/(distances+1e-7), logs) ) +\
                    _scalarmul(is_smooth_redescending, _scalarmul(a*(c-distances)/(c-b) * 1/(distances+1e-7) , logs)) +\
                    _scalarmul(is_constant_region, gs.zeros_like(logs)) )
        grad = _scalarmulsum(weights, grad) / sum_weights
        return loss, self.space.to_tangent(grad,base_point=base)
   
   
    def _riemannian_correntropy_loss_grad(self, points, base, critical_value=2.1105, weights=None, loss_and_grad=False):
        ''' correntropy loss
        grad_i = - (1/(C^3√(2π)))·e^{-ξ²/(2C²)} · v
        
        Parameters
        ----------
        points : array_like of shape [n, ...], data points.
        base: base point on manifold to get tangent space,
            and base point will be origin on the related tangent space.
        critical_value : float or array_like of shape [3,]
            cutoff in the loss function to control outliers
        weights : array_like of shape [n,], default None (equal weights)
            optional, default: None
        loss_and_grad: bool
            True only for computing loss.
            False when computing both loss and gradient of loss.
        
        Returns
        -------
        loss : float
            Weighted loss from the points provided.
        gradient : array-like, shape=[dim,]
            gradient of loss from the points provided.
            This should be on the tangent space of the manifold(space) provided.
        '''
        n = len(points)
        weights,sum_weights = self._set_weights(n,weights=None)
        self.critical_value = critical_value
        c = self.critical_value
       
        logs  = self.space.metric.log(point=points, base_point=base)
        distances = self.space.metric.norm(logs, base_point=base)
   
        norm_const = 1/(c * gs.sqrt(2 * gs.pi))
        loss = norm_const * (1 - gs.exp(- distances**2 / (2 * c**2)))
        loss = gs.sum(weights * loss) / sum_weights
        if not loss_and_grad:
            return loss
        grad_const = 1/(c**3 * gs.sqrt(2 * gs.pi))
        grad = _scalarmul( grad_const * gs.exp(- distances**2 / (2 * c**2)), logs )
        grad = _scalarmulsum(weights, grad) / sum_weights
        return loss, self.space.to_tangent(grad,base_point=base)
   
    def _riemannian_logistic_loss_grad(self, points, base, critical_value=1.205, weights=None, loss_and_grad=False):
        ''' logistic loss
                
        Parameters
        ----------
        points : array_like of shape [n, ...], data points.
        base: base point on manifold to get tangent space,
            and base point will be origin on the related tangent space.
        critical_value : float or array_like of shape [3,]
            cutoff in the loss function to control outliers
        weights : array_like of shape [n,], default None (equal weights)
            optional, default: None
        loss_and_grad: bool
            True only for computing loss.
            False when computing both loss and gradient of loss.
        
        Returns
        -------
        loss : float
            Weighted loss from the points provided.
        gradient : array-like, shape=[dim,]
            gradient of loss from the points provided.
            This should be on the tangent space of the manifold(space) provided.
        '''
        n = len(points)
        weights,sum_weights = self._set_weights(n,weights=None)
        self.critical_value = critical_value
        c = self.critical_value
   
        logs  = self.space.metric.log(point=points, base_point=base)
        distances = self.space.metric.norm(logs, base_point=base)
       
        loss = (c**2 * gs.log(gs.cosh(distances / c)))
        loss = gs.sum(weights * loss) / sum_weights
        if not loss_and_grad:
            return loss
        grad = _scalarmul( (c * gs.tanh(distances / c) / (distances+1e-7)), logs)
        grad = _scalarmulsum(weights, grad) / sum_weights
        return loss, self.space.to_tangent(grad,base_point=base)
       
   
    # ── #23 Lorentzian-quadratic loss ────────────────────────────────────────
    def _riemannian_lorentzian_loss_grad(self, points, base, critical_value=2.678, weights=None, loss_and_grad=False):
        ''' Lorentzian loss
                
        Parameters
        ----------
        points : array_like of shape [n, ...], data points.
        base: base point on manifold to get tangent space,
            and base point will be origin on the related tangent space.
        critical_value : float or array_like of shape [3,]
            cutoff in the loss function to control outliers
        weights : array_like of shape [n,], default None (equal weights)
            optional, default: None
        loss_and_grad: bool
            True only for computing loss.
            False when computing both loss and gradient of loss.
        
        Returns
        -------
        loss : float
            Weighted loss from the points provided.
        gradient : array-like, shape=[dim,]
            gradient of loss from the points provided.
            This should be on the tangent space of the manifold(space) provided.
        '''
        n = len(points)
        weights,sum_weights = self._set_weights(n,weights=None)
        self.critical_value = critical_value
        c = self.critical_value
   
        logs  = self.space.metric.log(point=points, base_point=base)
        distances = self.space.metric.norm(logs, base_point=base)
   
        t = distances**2 / (2 * c**2)
        loss = (1 - 1/(1 + t))
        loss = gs.sum(weights * loss) / sum_weights
        if not loss_and_grad:
            return loss
        grad = _scalarmul( 1 / (c**2 * (1 + distances**2/(2*c**2))**2 ) , logs )
        grad = _scalarmulsum(weights, grad) / sum_weights
        return loss, self.space.to_tangent(grad,base_point=base)

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
            #"batch": BatchGradientDescent,
        }
        self.optimizer = MAP_OPTIMIZER[value]()
        if value in ['autograd']:
            self.set(autograd = True)

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
       
        if self.method not in ['autograd']:
            if self.fun_provided:
                raise ValueError(f"Only autograd method available if Loss Function/M-estimator provided. check gs.has_autodiff() is True")
            self.estimate_ = self.optimizer.minimize(
                space=self.space,
                points=X,
                critical_value=self.critical_value,
                weights=weights,
                loss_grad_fun=self._set_m_estimator_loss(),
                init_point_method=self.init_point_method,
            )
            return self
        else:
            self.points = X
            self.weights = weights
            self._set_loss()
            self.estimate_ = self.optimizer.minimize(
                space=self.space,
                points=X,
                fun=self.loss_with_base,
                weights=weights,
                init_point_method=self.init_point_method,
            )


