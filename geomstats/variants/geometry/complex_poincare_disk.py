"""Variants of the metric on the complex Poincare disk."""

from geomstats.geometry.hermitian import RealCoordinatesDiffeo
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.geometry.scalar_product_metric import ScalarProductMetric


class PullbackComplexPoincareDiskMetric(PullbackDiffeoMetric):
    """Metric on the complex Poincare disk as a pullback metric.

    Variant of
    :class:`geomstats.geometry.complex_poincare_disk.ComplexPoincareDiskMetric`
    without its closed forms: the geometry is pulled back from the
    two-dimensional Poincare ball, with its metric scaled by 1/4, through the
    real coordinates of the complex disk.

    Parameters
    ----------
    space : ComplexPoincareDisk
        Manifold.
    """

    def __init__(self, space):
        image_space = PoincareBall(dim=2)
        image_space.equip_with_metric(ScalarProductMetric(image_space, 0.25))
        super().__init__(
            space=space,
            diffeo=RealCoordinatesDiffeo(dim=1),
            image_space=image_space,
        )
