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
<<<<<<< 1622dc5173f84a9014ac936d097347507a0c8ed2
=======
        assert isinstance(dimension, int) or dimension == math.inf
        assert dimension > 0
>>>>>>> Rebase
        self.dimension = dimension

    def belongs(self, point):
        """
        Evaluate if a point belongs to the manifold.
        """
        raise NotImplementedError('belongs is not implemented.')

    def regularize(self, point):
        """
        Regularize a point to the canonical representation
        chosen for the manifold.
        """
        raise NotImplementedError('regularize is not implemented.')
