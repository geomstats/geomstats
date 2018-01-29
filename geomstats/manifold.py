"""
Base for differentiable manifolds.
"""


class Manifold(object):
    """Base class for differentiable manifolds."""

    def __init__(self, dimension):
        self.dimension = dimension

    def belongs(self, point):
        """Check if the point belongs to the manifold."""
        raise NotImplementedError('belongs is not implemented.')

    def regularize(self, point):
        """
        Regularizes the point's coordinates to the canonical representation
        chosen for this manifold.
        """
        raise NotImplementedError('regularize is not implemented.')
