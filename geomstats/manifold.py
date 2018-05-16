"""
Class for differentiable manifolds.
"""


class Manifold(object):
    """
    Class for differentiable manifolds.
    """

    def __init__(self, dimension):
        assert isinstance(dimension, int) and dimension > 0
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
        return point
