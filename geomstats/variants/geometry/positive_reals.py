"""Variants of the logarithmic metric on the positive reals."""

import math

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.open_interval import OpenInterval, OpenIntervalMetric
from geomstats.geometry.positive_reals import LogDiffeo, PositiveReals, PowerDiffeo
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric


class PullbackLogMetric(PullbackDiffeoMetric):
    """Logarithmic metric on the positive reals as a pullback metric.

    Variant of :class:`geomstats.geometry.positive_reals.LogMetric`
    without its closed forms: the Christoffel symbols and the curvature
    tensors are computed from the pullback metric matrix with automatic
    differentiation.

    Parameters
    ----------
    space : PositiveReals
        Manifold.
    """

    def __init__(self, space):
        super().__init__(
            space=space,
            diffeo=LogDiffeo(),
            image_space=Euclidean(dim=1),
        )


class PullbackPowerMetric(PullbackDiffeoMetric):
    """Power-Euclidean metric on the positive reals as a pullback metric.

    Variant of :class:`geomstats.geometry.positive_reals.PowerMetric`
    without its closed forms: the Christoffel symbols and the curvature
    tensors are computed from the pullback metric matrix with automatic
    differentiation.

    Parameters
    ----------
    space : PositiveReals
        Manifold.
    power : float
        Non-zero power.
    """

    def __init__(self, space, power):
        if power > 0:
            image_space = PositiveReals(equip=False)
        else:
            image_space = OpenInterval(-math.inf, 0.0, equip=False)
        image_space.equip_with_metric(OpenIntervalMetric)

        super().__init__(
            space=space, diffeo=PowerDiffeo(power), image_space=image_space
        )
        self.power = power


class LogMetric(RiemannianMetric):
    """Logarithmic metric on the positive reals given only by its metric matrix.

    The Christoffel symbols and the curvature tensors are computed from the
    metric matrix with automatic differentiation, instead of the closed forms
    of :class:`geomstats.geometry.positive_reals.LogMetric`.
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
        return gs.expand_dims(1 / base_point**2, axis=-1)


class PowerMetric(RiemannianMetric):
    """Power-Euclidean metric on the positive reals given by its metric matrix.

    The Christoffel symbols and the curvature tensors are computed from the
    metric matrix with automatic differentiation, instead of the closed forms
    of :class:`geomstats.geometry.positive_reals.PowerMetric`.

    Parameters
    ----------
    space : PositiveReals
        Manifold.
    power : float
        Non-zero power.
    """

    def __init__(self, space, power):
        super().__init__(space)
        self.power = power

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
        return gs.expand_dims(base_point ** (2 * self.power - 2), axis=-1)
