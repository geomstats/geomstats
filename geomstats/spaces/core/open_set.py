"""Abstract class for open sets.

Lead authors: Nicolas Guigui and Nina Miolane.
"""

import abc
import geomstats.backend as gs
from geomstats.spaces.core import Manifold


class OpenSet(Manifold):
    """Class for manifolds that are open sets of a vector space.

    In this case, tangent vectors are identified with vectors of the embedding
    space.

    Parameters
    ----------
    dim: int
        Dimension of the manifold. It is often the same as the embedding space
        dimension but may differ in some cases.
    embedding_space: VectorSpace
        Embedding space that contains the manifold.
    """

    def __init__(self, dim, embedding_space, **kwargs):
        kwargs.setdefault("shape", embedding_space.shape)
        super().__init__(dim=dim, **kwargs)
        self.embedding_space = embedding_space

    def is_tangent(self, vector, base_point=None, atol=gs.atol):
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
        return self.embedding_space.belongs(vector, atol)

    def to_tangent(self, vector, base_point=None):
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
        return self.embedding_space.projection(vector)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample random points on the manifold.

        If the manifold is compact, a uniform distribution is used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample for non compact manifolds.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., {dim, [n, n]}]
            Points sampled on the hypersphere.
        """
        sample = self.embedding_space.random_point(n_samples, bound)
        return self.projection(sample)

    @abc.abstractmethod
    def projection(self, point):
        """Project a point in embedding manifold on manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., dim]
            Projected point.
        """
