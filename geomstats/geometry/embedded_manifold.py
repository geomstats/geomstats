"""Manifold embedded in another manifold."""

from geomstats.geometry.manifold import Manifold


class EmbeddedManifold(Manifold):
    """Class for manifolds embedded in an embedding manifold.

    Parameters
    ----------
    dim : int
        Dimension of the embedded manifold.
    embedding_manifold : Manifold
        Embedding manifold.
    default_point_type : str, {'vector', 'matrix'}
        Point type.
        Optional, default: 'vector'.
    default_coords_type : str, {'intrinsic', 'extrinsic', etc}
        Coordinate type.
        Optional, default: 'intrinsic'.
    """

    def __init__(self, dim, embedding_manifold, default_point_type='vector',
                 default_coords_type='intrinsic'):
        super(EmbeddedManifold, self).__init__(
            dim=dim, default_point_type=default_point_type,
            default_coords_type=default_coords_type)
        self.embedding_manifold = embedding_manifold

    def intrinsic_to_extrinsic_coords(self, point_intrinsic):
        """Convert from intrinsic to extrinsic coordinates.

        Parameters
        ----------
        point_intrinsic : array-like, shape=[..., dim]
            Point in the embedded manifold in intrinsic coordinates.

        Returns
        -------
        point_extrinsic : array-like, shape=[..., dim_embedding]
            Point in the embedded manifold in extrinsic coordinates.
        """
        raise NotImplementedError(
            'intrinsic_to_extrinsic_coords is not implemented.')

    def extrinsic_to_intrinsic_coords(self, point_extrinsic):
        """Convert from extrinsic to intrinsic coordinates.

        Parameters
        ----------
        point_extrinsic : array-like, shape=[..., dim_embedding]
            Point in the embedded manifold in extrinsic coordinates,
            i. e. in the coordinates of the embedding manifold.

        Returns
        -------
        point_intrinsic : array-lie, shape=[..., dim]
            Point in the embedded manifold in intrinsic coordinates.
        """
        raise NotImplementedError(
            'extrinsic_to_intrinsic_coords is not implemented.')

    def projection(self, point):
        """Project a point in embedding manifold on embedded manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim_embedding]
            Point in embedding manifold.

        Returns
        -------
            Projected point.
        """
        raise NotImplementedError(
            'projection is not implemented.')
