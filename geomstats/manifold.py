"""
Manifold, i.e. a topological space that locally resembles
Euclidean space near each point.
"""

import math


class Manifold(object):
    """
    Class for manifolds.
    """

    def __init__(self, dimension):

        assert isinstance(dimension, int) or dimension == math.inf
        assert dimension > 0

        self.dimension = dimension

    def belongs(self, point, point_type=None):
        """
        Evaluate if a point belongs to the manifold.
        """
        raise NotImplementedError('belongs is not implemented.')

    def regularize(self, point, point_type=None):
        """
        Regularize a point to the canonical representation
        chosen for the manifold.
        """
        return point
