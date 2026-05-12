"""Exponential barycenter.

Lead author: Nicolas Guigui.
"""

import logging

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.lie_group import LieGroup, MatrixLieGroup
from geomstats.learning.frechet_mean import (
    BaseGradientDescent,
    LinearMean,
    _BaseMeanEstimator,
    _scalarmulsum,
)


class GroupGradientDescent(BaseGradientDescent):
    """Gradient descent for exponential barycenter."""

    def minimize(self, space, points, weights=None):
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

        Returns
        -------
        exp_bar : array-like, shape=[dim, dim]
            Exponential barycenter of the input points.
        """
        n_samples = points.shape[0]

        if n_samples == 1:
            return points[0]

        if weights is None:
            weights = gs.ones((n_samples,))

        sum_weights = gs.sum(weights)

        mean = points[0] if self.init_point is None else self.init_point

        grad_norm = 0.0

        for iteration in range(self.max_iter):
            if not (grad_norm > self.epsilon or iteration == 0):
                break

            inv_mean = space.inverse(mean)
            centered_points = space.compose(inv_mean, points)

            logs = space.log(point=centered_points)
            tangent_mean = self.init_step_size * gs.einsum(
                "n, nk...->k...", weights / sum_weights, logs
            )
            mean_next = space.compose(mean, space.exp(tangent_vec=tangent_mean))

            grad_norm = gs.linalg.norm(tangent_mean)

            mean = mean_next

        else:
            logging.warning(
                f"Maximum number of iterations {self.max_iter} reached. "
                "The mean may be inaccurate"
            )

        if self.verbose:
            logging.info(f"n_iter: {iteration}, final gradient norm: {grad_norm}")
        return mean


class GroupBarycenter(_BaseMeanEstimator):
    """Empirical exponential barycenter for matrix groups.

    Parameters
    ----------
    space : LieGroup
        Lie group instance on which the data lie.
    optimizer_kwargs : dict or None
        Keyword arguments passed to the optimizer constructor.
        Optional, default: None.

    Attributes
    ----------
    estimate_ : array-like, shape=[dim, dim]
        If fit, exponential barycenter.
    optimizer_ : object
        Optimizer instance used during fitting.
    """

    # TODO (nguigs): use closed form expression for special euclidean
    #  group as before PR #537

    def __init__(self, space, optimizer_kwargs=None):
        super().__init__(space, optimizer_kwargs=optimizer_kwargs)

    def _make_optimizer(self):
        return GroupGradientDescent(**self.optimizer_kwargs)


class ConnectionGradientDescent(BaseGradientDescent):
    """Gradient descent for exponential barycenter."""

    def minimize(self, space, points, weights=None):
        """Perform default gradient descent."""
        n_points = gs.shape(points)[0]
        if weights is None:
            weights = gs.ones((n_points,))

        mean = points[0] if self.init_point is None else self.init_point

        if n_points == 1:
            return mean

        sum_weights = gs.sum(weights)

        norm_old = gs.inf
        step_size = self.init_step_size

        for _ in range(self.max_iter):
            logs = space.connection.log(point=points, base_point=mean)

            tangent_mean = _scalarmulsum(weights, logs)
            tangent_mean /= sum_weights

            if gs.amax(gs.abs(tangent_mean)) < self.epsilon:
                break

            new_mean = space.connection.exp(step_size * tangent_mean, mean)

            norm = gs.linalg.norm(tangent_mean)
            if norm < norm_old:
                mean = new_mean
                norm_old = norm
            elif norm > norm_old:
                step_size = step_size / 2.0

        else:
            logging.warning(
                "Maximum number of iterations %d reached. The mean may be inaccurate",
                self.max_iter,
            )

        return mean


class GeneralExponentialBarycenter(_BaseMeanEstimator):
    """Exponential barycenter.

    Parameters
    ----------
    space : Manifold
        Equipped manifold on which the data lie.
    optimizer_kwargs : dict or None
        Keyword arguments passed to the optimizer constructor.
        Optional, default: None.

    Attributes
    ----------
    estimate_ : array-like, shape=[*space.shape]
        Estimated mean after calling `fit`.
    optimizer_ : object
        Optimizer instance used during fitting.
    """

    def _make_optimizer(self):
        return ConnectionGradientDescent(**self.optimizer_kwargs)


def ExponentialBarycenter(space, *args, **kwargs):
    """Exponential barycenter."""
    if isinstance(space, Euclidean):
        Estimator = LinearMean

    elif isinstance(space, (MatrixLieGroup, LieGroup)):
        Estimator = GroupBarycenter

    else:
        Estimator = GeneralExponentialBarycenter

    return Estimator(space, *args, **kwargs)
