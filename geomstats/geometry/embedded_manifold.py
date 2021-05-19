"""Manifold embedded in another manifold."""
from abc import ABC, abstractmethod

import geomstats.backend as gs
from geomstats.geometry.manifold import Manifold


class EmbeddedManifold(Manifold, ABC):
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

    def __init__(self, dim, embedding_manifold, submersion, tangent_submersion,
                 default_point_type='vector',
                 default_coords_type='intrinsic', **kwargs):
        super(EmbeddedManifold, self).__init__(
            dim=dim, default_point_type=default_point_type,
            default_coords_type=default_coords_type, **kwargs)
        self.embedding_manifold = embedding_manifold
        self.embedding_metric = embedding_manifold.metric
        self.submersion = submersion
        self.tangent_submersion = tangent_submersion

    def belongs(self, point, atol=gs.atol):
        belongs = self.embedding_manifold.belongs(point, atol)
        constraint = gs.isclose(self.submersion(point), 0., atol=atol)
        if self.default_point_type == 'matrix':
            constraint = gs.all(constraint, axis=(-2, -1))
        return gs.logical_and(belongs, constraint)

    def is_tangent(self, vector, base_point, atol=gs.atol):
        tangent_sub_applied = self.tangent_submersion(vector, base_point)
        constraint = gs.isclose(tangent_sub_applied, 0., atol=atol)
        if self.default_point_type == 'matrix':
            constraint = gs.all(constraint, axis=(-2, -1))
        return constraint

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

    @abstractmethod
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
        pass


class OpenSet(Manifold, ABC):
    def __init__(self, dim, ambient_manifold, **kwargs):
        super().__init__(dim, **kwargs)
        self.ambient_manifold = ambient_manifold

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        return self.ambient_manifold.belongs(vector, atol)

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        return self.ambient_manifold.projection(vector)

    def random_point(self, n_samples=1, bound=1.):
        sample = self.ambient_manifold.random_point(n_samples, bound)
        return self.projection(sample)

    @abstractmethod
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
        pass
