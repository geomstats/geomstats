"""Manifold embedded in another manifold."""

import math

from geomstats.geometry.manifold import Manifold


class EmbeddedManifold(Manifold):
    """Class for manifolds embedded in an embedding manifold.

    Parameters
    ----------
    dimension : int
        Dimension of the embedded manifold.
    embedding_manifold : Manifold
        Embedding manifold.
    """

    def __init__(self, dimension, embedding_manifold):
        assert isinstance(dimension, int) or dimension == math.inf
        assert dimension > 0
        super(EmbeddedManifold, self).__init__(
            dimension=dimension)
        self.embedding_manifold = embedding_manifold

    def intrinsic_to_extrinsic_coords(self, point_intrinsic):
        """Convert from intrinsic to extrinsic coordinates.

        Parameters
        ----------
        point_intrinsic : array-like, shape=[n_samples, dim]
            Point in the embedded manifold in intrinsic coordinates.
        """
        raise NotImplementedError(
            'intrinsic_to_extrinsic_coords is not implemented.')

    def extrinsic_to_intrinsic_coords(self, point_extrinsic):
        """Convert from extrinsic to intrinsic coordinates.

        Parameters
        ----------
        point_extrinsic : array-like, shape=[n_samples, dim_embedding]
            Point in the embedded manifold in extrinsic coordinates,
            i. e. in the coordinates of the embedding manifold.
        """
        raise NotImplementedError(
            'extrinsic_to_intrinsic_coords is not implemented.')

    def projection(self, point):
        """Project a point in embedding manifold on embedded manifold.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dim_embedding]
            Point in embedding manifold
        """
        raise NotImplementedError(
            'projection is not implemented.')

    def projection_to_tangent_space(self, vector, base_point):
        """Project a vector to a tangent space of the embedded manifold.

        Parameters
        ----------
        vector : array-like, shape=[n_samples, dim_embedding]
            Vector at the tangent space of the embedding manifold.
        base_point : array-like, shape=[n_samples, dim_embedding]
            Point on the embedded manifold, in extrinsic coordinates.
        """
        raise NotImplementedError(
            'projection_to_tangent_space is not implemented.')
