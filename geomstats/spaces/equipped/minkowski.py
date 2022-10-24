"""Minkowski space."""

from geomstats.spaces.coordinatized import RealVectorSpace
from geomstats.structure.metric import MinkowskiMetric


class Minkowski(RealVectorSpace):
    """Class for Minkowski space.

    This is a real vector space space endowed with the inner-product of signature (
    dim-1, 1).

    Parameters
    ----------
    dim : int
       Dimension of Minkowski space.
    """

    def __init__(self, dim, equip_default=True):
        super().__init__(
            dim=dim,
            shape=(dim,),
            equip_default=equip_default
        )

    def _default_metric(self):
        Metric = MinkowskiMetric
        kwargs = {}
        return Metric, kwargs
