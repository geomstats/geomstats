"""
Manifold embedded in another manifold.
"""

from geomstats.manifold import Manifold


class EmbeddedManifold(Manifold):
    """
    Class for manifolds embedded in another manifold.
    """

    def __init__(self, dimension, embedding_manifold):
        assert isinstance(dimension, int) and dimension > 0
        super(EmbeddedManifold, self).__init__(
            dimension=dimension)
        self.embedding_manifold = embedding_manifold

    def intrinsic_to_extrinsic_coords(self, point_intrinsic):
        raise NotImplementedError(
            'intrinsic_to_extrinsic_coords is not implemented.')

    def extrinsic_to_intrinsic_coords(self, point_extrinsic):
        raise NotImplementedError(
            'extrinsic_to_intrinsic_coords is not implemented.')

    def projection(self, point):
        raise NotImplementedError(
            'projection is not implemented.')

    def projection_to_tangent_space(self, vector, base_point):
        raise NotImplementedError(
            'projection_to_tangent_space is not implemented.')
