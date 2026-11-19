"""Variants of the metric on the Poincare half-space."""

from geomstats.geometry._hyperbolic import HyperbolicDiffeo
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric


class PullbackPoincareHalfSpaceMetric(PullbackDiffeoMetric):
    """Hyperbolic metric on the Poincare half-space as a pullback metric.

    Variant of
    :class:`geomstats.geometry.poincare_half_space.PoincareHalfSpaceMetric`
    without its closed forms: the Christoffel symbols and the curvature
    tensors are computed from the pullback metric matrix with automatic
    differentiation.

    Parameters
    ----------
    space : PoincareHalfSpace
        Manifold.
    """

    def __init__(self, space):
        image_space = PoincareBall(dim=space.dim)
        super().__init__(
            space=space,
            diffeo=HyperbolicDiffeo(space.coords_type, image_space.coords_type),
            image_space=image_space,
        )
