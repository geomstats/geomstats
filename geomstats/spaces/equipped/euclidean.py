"""Euclidean space."""

import geomstats.backend as gs
from geomstats.spaces.coordinatized import RealVectorSpace
from geomstats.structure.metric import EuclideanMetric


class Euclidean(RealVectorSpace):
    """Class for Euclidean spaces.

    By definition, a Euclidean space is a vector space of a given
    dimension, equipped with a Euclidean metric.

    Parameters
    ----------
    dim : int
        Dimension of the Euclidean space.
    """

    def __init__(self, dim, equip_default=True):
        super().__init__(
            dim=dim,
            shape=(dim,),
            equip_default=equip_default,
        )

    def _default_metric(self):
        Metric = EuclideanMetric
        kwargs = {}
        return Metric, kwargs
