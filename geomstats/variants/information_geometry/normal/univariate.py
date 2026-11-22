"""Variants of the Fisher-Rao metric on univariate normal distributions."""

import math

import geomstats.backend as gs
from geomstats.geometry.euclidean import CoordinateScalingDiffeo
from geomstats.geometry.poincare_half_space import PoincareHalfSpace
from geomstats.geometry.pullback_metric import CoordinatePullbackDiffeoMetric
from geomstats.geometry.scalar_product_metric import ScalarProductMetric


class PullbackUnivariateNormalMetric(CoordinatePullbackDiffeoMetric):
    """Fisher-Rao metric on univariate normal distributions as a pullback metric.

    Variant of
    :class:`geomstats.information_geometry.normal.univariate.UnivariateNormalMetric`
    without its closed forms: the Christoffel symbols and the curvature
    tensors are computed from the pullback metric matrix with automatic
    differentiation.

    Parameters
    ----------
    space : UnivariateNormalDistributions
        Manifold.
    """

    def __init__(self, space):
        super().__init__(
            space,
            diffeo=CoordinateScalingDiffeo(gs.array([1 / math.sqrt(2.0), 1.0])),
            image_space=PoincareHalfSpace(dim=2).equip_with_metric(
                ScalarProductMetric, scale=2.0
            ),
        )
