"""Exponential barycenter.

Lead author: Nicolas Guigui.
"""

import logging

from sklearn.base import BaseEstimator

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.learning.frechet_mean import BaseGradientDescent, LinearMean


class GradientDescent(BaseGradientDescent):
    """Gradient descent for exponential barycenter."""

    def minimize(self, group, points, weights=None):
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

            inv_mean = group.inverse(mean)
            centered_points = group.compose(inv_mean, points)

            logs = group.log(point=centered_points)
            tangent_mean = self.init_step_size * gs.einsum(
                "n, nk...->k...", weights / sum_weights, logs
            )
            mean_next = group.compose(mean, group.exp(tangent_vec=tangent_mean))

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


class ExponentialBarycenter(BaseEstimator):
    """Empirical exponential barycenter for matrix groups.

    Parameters
    ----------
    space : LieGroup
        Lie group instance on which the data lie.

    Attributes
    ----------
    estimate_ : array-like, shape=[dim, dim]
        If fit, exponential barycenter.
    """

    def __new__(cls, space):
        """Interface for instantiating proper algorithm."""
        if isinstance(space, Euclidean):
            return LinearMean(space)

        return super().__new__(cls)

    def __init__(self, space):
        self.space = space

        self.optimizer = GradientDescent()

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
        # TODO (nguigs): use closed form expression for special euclidean
        #  group as before PR #537

        self.estimate_ = self.optimizer.minimize(
            group=self.space,
            points=X,
            weights=weights,
        )

        return self
