"""Euclidean space."""

import geomstats.backend as gs
from geomstats.geometry.base import VectorSpace
from geomstats.geometry.riemannian_metric import RiemannianMetric


class Euclidean(VectorSpace):
    """Class for Euclidean spaces.

    By definition, a Euclidean space is a vector space of a given
    dimension, equipped with a Euclidean metric.

    Parameters
    ----------
    dim : int
        Dimension of the Euclidean space.
    """

    def __init__(self, dim):
        super(Euclidean, self).__init__(
            shape=(dim,),
            metric=EuclideanMetric(dim, shape=(dim,)),
        )

    def get_identity(self):
        """Get the identity of the group.

        Returns
        -------
        identity : array-like, shape=[n]
        """
        identity = gs.zeros(self.dim)
        return identity

    identity = property(get_identity)

    def _create_basis(self):
        """Create the canonical basis."""
        return gs.eye(self.dim)

    def exp(self, tangent_vec, base_point=None):
        """Compute the group exponential, which is simply the addition.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n]
            Point from which the exponential is computed.

        Returns
        -------
        point : array-like, shape=[..., n]
            Group exponential.
        """
        if not self.belongs(tangent_vec):
            raise ValueError("The update must be of the same dimension")
        return tangent_vec + base_point


class EuclideanMetric(RiemannianMetric):
    """Class for Euclidean metrics.

    As a Riemannian metric, the Euclidean metric is:

    - flat: the inner-product is independent of the base point;
    - positive definite: it has signature (dimension, 0, 0),
      where dimension is the dimension of the Euclidean space.

    Parameters
    ----------
    dim : int
        Dimension of the Euclidean space.
    """

    def __init__(self, dim, shape=None):
        super(EuclideanMetric, self).__init__(
            dim=dim,
            shape=shape,
            signature=(dim, 0),
        )

    def metric_matrix(self, base_point=None):
        """Compute the inner-product matrix, independent of the base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        inner_prod_mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        mat = gs.eye(self.dim)
        return mat

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Inner product between two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a: array-like, shape=[..., dim]
            Tangent vector at base point.
        tangent_vec_b: array-like, shape=[..., dim]
            Tangent vector at base point.
        base_point: array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """
        return gs.dot(tangent_vec_a, tangent_vec_b)

    def norm(self, vector, base_point=None):
        """Compute norm of a vector.

        Norm of a vector associated to the inner product
        at the tangent space at a base point.

        Note: This only works for positive-definite
        Riemannian metrics and inner products.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        norm : array-like, shape=[...,]
            Norm.
        """
        return gs.linalg.norm(vector, axis=-1)

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute exp map of a base point in tangent vector direction.

        The Riemannian exponential is vector addition in the Euclidean space.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            Riemannian exponential.
        """
        exp = base_point + tangent_vec
        return exp

    def log(self, point, base_point, **kwargs):
        """Compute log map using a base point and other point.

        The Riemannian logarithm is the subtraction in the Euclidean space.

        Parameters
        ----------
        point: array-like, shape=[..., dim]
            Point.
        base_point: array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        log: array-like, shape=[..., dim]
            Riemannian logarithm.
        """
        log = point - base_point
        return log

    def parallel_transport(
        self, tangent_vec, base_point=None, direction=None, end_point=None
    ):
        r"""Compute the parallel transport of a tangent vector.

        On a Euclidean space, the parallel transport of a (tangent) vector
        returns the vector itself.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., dim]
            Point on the manifold. Point to transport from.
            Optional, default: None
        direction : array-like, shape=[..., dim]
            Tangent vector at base point, along which the parallel transport
            is computed.
            Optional, default: None.
        end_point : array-like, shape=[..., dim]
            Point on the manifold. Point to transport to.
            Optional, default: None.

        Returns
        -------
        transported_tangent_vec: array-like, shape=[..., dim]
            Transported tangent vector at `exp_(base_point)(tangent_vec_b)`.
        """
        return tangent_vec
