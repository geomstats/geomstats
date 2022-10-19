"""Exponential barycenter.

Lead author: Nicolas Guigui.
"""

import logging

from sklearn.base import BaseEstimator

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.learning.frechet_mean import linear_mean

EPSILON = 1e-6


def _default_gradient_descent(
    group,
    points,
    weights=None,
    max_iter=32,
    init_step_size=1.0,
    epsilon=EPSILON,
    verbose=False,
):
    """Compute the (weighted) group exponential barycenter of `points`.

    Parameters
    ----------
    group : LieGroup
        Instance of the class LieGroup.
    points : array-like, shape=[n_samples, dim, dim]
        Input points lying in the Lie Group.
    weights : array-like, shape=[n_samples,]
        Weights associated to the points.
        Optional, defaults to 1 for each point if None.
    max_iter : int
        Maximum number of iterations to perform in the gradient descent.
        Optional, default: 32.
    epsilon : float
        Tolerance to reach convergence. The exstrinsic norm of the
        gradient is used as criterion.
        Optional, default: 1e-6.
    init_step_size : float
        Learning rate in the gradient descent.
        Optional, default: 1.
    verbose : bool
        Level of verbosity to inform about convergence.
        Optional, default: False.

    Returns
    -------
    exp_bar : array-like, shape=[dim, dim]
        Exponential barycenter of the input points.
    """
    ndim = 2 if group.default_point_type == "vector" else 3
    if gs.ndim(gs.array(points)) < ndim or len(points) == 1:
        return points[0] if len(points) == 1 else points

    n_points = points.shape[0]
    if weights is None:
        weights = gs.ones((n_points,))
    weights = gs.cast(weights, gs.float32)
    sum_weights = gs.sum(weights)

    mean = points[0]

    sq_dists_between_iterates = []
    grad_norm = 0.0

    for iteration in range(max_iter):
        if not (grad_norm > epsilon or iteration == 0):
            break
        inv_mean = group.inverse(mean)
        centered_points = group.compose(inv_mean, points)
        logs = group.log(point=centered_points)
        tangent_mean = init_step_size * gs.einsum(
            "n, nk...->k...", weights / sum_weights, logs
        )
        mean_next = group.compose(mean, group.exp(tangent_vec=tangent_mean))

        grad_norm = gs.linalg.norm(tangent_mean)
        sq_dists_between_iterates.append(grad_norm)

        mean = mean_next

    else:
        logging.warning(
            f"Maximum number of iterations {max_iter} reached. "
            "The mean may be inaccurate"
        )

    if verbose:
        logging.info(f"n_iter: {iteration}, final gradient norm: {grad_norm}")
    return mean


class ExponentialBarycenter(BaseEstimator):
    """Empirical exponential barycenter for matrix groups.

    Parameters
    ----------
    group : LieGroup
        Lie group instance on which the data lie.
    max_iter : int
        Maximum number of iterations to perform in the gradient descent.
        Optional, default: 32.
    epsilon : float
        Tolerance to reach convergence. The extrinsic norm of the
        gradient is used as criterion.
        Optional, default: 1e-6.
    init_step_size : float
        Learning rate in the gradient descent.
        Optional, default: 1.
    verbose : bool
        Level of verbosity to inform about convergence.
        Optional, default: 1.

    Attributes
    ----------
    estimate_ : array-like, shape=[dim, dim]
        If fit, exponential barycenter.
    """

    def __init__(
        self,
        group,
        max_iter=32,
        epsilon=EPSILON,
        init_step_size=1.0,
        point_type=None,  # TODO: undocumented and unused parameter
        verbose=False,
    ):
        self.group = group
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.verbose = verbose
        self.init_step_size = init_step_size
        self.point_type = point_type
        self.estimate_ = None

    def fit(self, X, y=None, weights=None):
        """Compute the empirical weighted exponential barycenter.

        Parameters
        ----------
        X : array-like, shape=[n_samples, dim, dim]
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
        if isinstance(self.group, Euclidean):
            mean = linear_mean(points=X, weights=weights)

        # TODO (nguigs): use closed form expression for special euclidean
        #  group as before PR #537

        else:
            mean = _default_gradient_descent(
                group=self.group,
                points=X,
                weights=weights,
                max_iter=self.max_iter,
                init_step_size=self.init_step_size,
                epsilon=self.epsilon,
                verbose=self.verbose,
            )
        self.estimate_ = mean

        return self
