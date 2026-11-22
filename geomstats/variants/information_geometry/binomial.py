"""Variants of the Fisher-Rao metric on binomial distributions."""

import math

import geomstats.backend as gs
from geomstats.geometry.open_interval import ArcsinSqrtDiffeo, OpenInterval
from geomstats.geometry.pullback_metric import CoordinatePullbackDiffeoMetric
from geomstats.geometry.riemannian_metric import CoordinateRiemannianMetric


class BinomialMetric(CoordinateRiemannianMetric):
    """Fisher-Rao metric on binomial distributions given by its metric matrix.

    The Christoffel symbols and the curvature tensors are computed from the
    metric matrix with automatic differentiation, instead of the closed forms
    of :class:`geomstats.information_geometry.binomial.BinomialMetric`.
    """

    def metric_matrix(self, base_point):
        """Compute the inner-product matrix.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., 1, 1]
            Inner-product matrix.
        """
        return gs.expand_dims(
            self._space.n_draws / (base_point * (1 - base_point)), axis=-1
        )


class PullbackBinomialMetric(CoordinatePullbackDiffeoMetric):
    """Fisher-Rao metric on binomial distributions as a pullback metric.

    Variant of :class:`geomstats.information_geometry.binomial.BinomialMetric`
    without its closed forms: the Christoffel symbols and the curvature
    tensors are computed from the pullback metric matrix with automatic
    differentiation.
    """

    def __init__(self, space):
        n_draws = space.n_draws
        super().__init__(
            space=space,
            diffeo=ArcsinSqrtDiffeo(n_draws),
            image_space=OpenInterval(0.0, math.pi * math.sqrt(n_draws)),
        )
