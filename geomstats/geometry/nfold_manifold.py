"""N-fold product manifold.

Lead author: Nicolas Guigui, John Harvey.
"""

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.vectorization import get_batch_shape


class NFoldManifold(Manifold):
    r"""Class for an n-fold product manifold :math:`M^n`.

    Define a manifold as the product manifold of n copies of a given base
    manifold M.

    Parameters
    ----------
    base_manifold : Manifold
        Base manifold.
    n_copies : int
        Number of replication of the base manifold.
    """

    def __init__(
        self,
        base_manifold,
        n_copies,
        equip=True,
    ):
        geomstats.errors.check_integer(n_copies, "n_copies")
        dim = n_copies * base_manifold.dim
        shape = (n_copies,) + base_manifold.shape

        self.base_manifold = base_manifold
        self.n_copies = n_copies

        super().__init__(
            dim=dim,
            shape=shape,
            intrinsic=base_manifold.intrinsic,
            equip=equip,
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return NFoldMetric

    def belongs(self, point, atol=gs.atol):
        """Test if a point belongs to the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., n_copies, *base_shape]
            Point.
        atol : float,
            Tolerance.

        Returns
        -------
        belongs : array-like, shape=[..., n_copies, *base_shape]
            Boolean evaluating if the point belongs to the manifold.
        """
        batch_shape = get_batch_shape(self.point_ndim, point)
        point_ = gs.reshape(point, (-1, *self.base_manifold.shape))

        each_belongs = self.base_manifold.belongs(point_, atol=atol)

        reshaped = gs.reshape(each_belongs, batch_shape + (self.n_copies,))
        return gs.all(reshaped, axis=-1)

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        The tangent space of the product manifold is the direct sum of
        tangent spaces.

        Parameters
        ----------
        vector : array-like, shape=[..., n_copies, *base_shape]
            Vector.
        base_point : array-like, shape=[..., n_copies, *base_shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        vector_, base_point_ = gs.broadcast_arrays(vector, base_point)
        batch_shape = get_batch_shape(self.point_ndim, vector_)
        base_point_ = gs.reshape(base_point_, (-1, *self.base_manifold.shape))
        vector_ = gs.reshape(vector_, (-1, *self.base_manifold.shape))

        each_tangent = self.base_manifold.is_tangent(vector_, base_point_, atol=atol)

        reshaped = gs.reshape(each_tangent, batch_shape + (self.n_copies,))
        return gs.all(reshaped, axis=-1)

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        The tangent space of the product manifold is the direct sum of
        tangent spaces.

        Parameters
        ----------
        vector : array-like, shape=[..., n_copies, *base_shape]
            Vector.
        base_point : array-like, shape=[..., n_copies, *base_shape]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n_copies, *base_shape]
            Tangent vector at base point.
        """
        vector_, base_point_ = gs.broadcast_arrays(vector, base_point)
        base_point_ = gs.reshape(base_point_, (-1, *self.base_manifold.shape))
        batch_shape = get_batch_shape(self.point_ndim, vector_)
        vector_ = gs.reshape(vector_, (-1, *self.base_manifold.shape))

        each_tangent = self.base_manifold.to_tangent(vector_, base_point_)

        return gs.reshape(each_tangent, batch_shape + self.shape)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in the product space from the product distribution.

        The distribution used is the product of the distributions that each copy of the
        manifold uses in its own random_point method.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples.
        bound : float
            Bound of the interval in which to sample for non compact manifolds.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n_copies, *base_shape]
            Points sampled on the product manifold.
        """
        sample = self.base_manifold.random_point(n_samples * self.n_copies, bound)
        reshaped = gs.reshape(
            sample, (n_samples, self.n_copies) + self.base_manifold.shape
        )
        if n_samples > 1:
            return reshaped
        return gs.squeeze(reshaped, axis=0)

    def projection(self, point):
        """Project a point from product embedding manifold to the product manifold.

        Parameters
        ----------
        point : array-like, shape=[..., n_copies, *base_shape]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., n_copies, *base_shape]
            Projected point.
        """
        if hasattr(self.base_manifold, "projection"):
            batch_shape = get_batch_shape(self.point_ndim, point)
            point_ = gs.reshape(point, (-1, *self.base_manifold.shape))
            projected = self.base_manifold.projection(point_)
            return gs.reshape(projected, batch_shape + self.shape)
        raise NotImplementedError(
            "The base manifold does not implement a projection method."
        )


class NFoldMetric(RiemannianMetric):
    r"""Class for an n-fold product manifold :math:`M^n`.

    Define a manifold as the product manifold of n copies of a given base
    manifold M.

    Parameters
    ----------
    space : NFoldManifold
        Base space.
    scales : array-like
        Scale of each metric in the product.
    """

    def __init__(self, space, scales=None, signature=None):
        if scales is not None:
            for scale in scales:
                geomstats.errors.check_positive(scale, "Each value in scales")

            if len(scales) != space.n_copies:
                raise ValueError(
                    "Number of scales should be equal to number of factors"
                )
        self.scales = scales

        super().__init__(space=space, signature=signature)

    def metric_matrix(self, base_point):
        """Compute the matrix of the inner-product.

        Matrix of the inner-product defined by the Riemmanian metric
        at point base_point of the manifold.

        Parameters
        ----------
        base_point : array-like, shape=[..., n_copies, *base_shape]
            Point on the manifold at which to compute the inner-product matrix.

        Returns
        -------
        matrix : array-like, shape=[..., n_copies, dim, dim]
            Matrix of the inner-product at the base point.
        """
        base_manifold = self._space.base_manifold
        batch_shape = get_batch_shape(self._space.point_ndim, base_point)

        point_ = gs.reshape(base_point, (-1, *base_manifold.shape))
        matrices = base_manifold.metric.metric_matrix(point_)

        dim = base_manifold.shape[-1]
        reshaped = gs.reshape(matrices, batch_shape + (self._space.n_copies, dim, dim))

        if self.scales is not None:
            reshaped = gs.einsum("j,...jkl->...jkl", self.scales, reshaped)

        return reshaped

    def pointwise_inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Pointwise inner product.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n_copies, *base_shape]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., n_copies, *base_shape]
            Second tangent vector at base point.
        base_point : array-like, shape=[..., n_copies, *base_shape]
            Point on the manifold.
            Optional, default: None.

        Returns
        -------
        pointwise_inner_prod : array-like, shape=[..., n_copies]
            Inner-product of the two tangent vectors.
        """
        base_manifold = self._space.base_manifold

        tangent_vec_a_, tangent_vec_b_, point_ = gs.broadcast_arrays(
            tangent_vec_a, tangent_vec_b, base_point
        )
        batch_shape = get_batch_shape(self._space.point_ndim, tangent_vec_a_)

        point_ = gs.reshape(point_, (-1, *base_manifold.shape))
        vector_a = gs.reshape(tangent_vec_a_, (-1, *base_manifold.shape))
        vector_b = gs.reshape(tangent_vec_b_, (-1, *base_manifold.shape))
        inner_each = base_manifold.metric.inner_product(vector_a, vector_b, point_)

        reshaped = gs.reshape(inner_each, batch_shape + (self._space.n_copies,))

        if self.scales is not None:
            reshaped = gs.einsum("j,...j->...j", self.scales, reshaped)

        return reshaped

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the inner-product of two tangent vectors at a base point.

        Inner product defined by the Riemannian metric at point `base_point`
        between tangent vectors `tangent_vec_a` and `tangent_vec_b`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n_copies, *base_shape]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., n_copies, *base_shape]
            Second tangent vector at base point.
        base_point : array-like, shape=[..., n_copies, *base_shape]
            Point on the manifold.
            Optional, default: None.

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Inner-product of the two tangent vectors.
        """
        return gs.sum(
            self.pointwise_inner_product(tangent_vec_a, tangent_vec_b, base_point),
            axis=-1,
        )

    def pointwise_norm(self, tangent_vec, base_point):
        """Compute the pointwise norms of a tangent vector.

        Compute the norms of the components of a tangent vector at the different
        sampling points of a base curve.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n_copies, *base_shape]
            Tangent vector to discrete curve.
        base_point : array-like, shape=[..., n_copies, *base_shape]
            Point representing a discrete curve.

        Returns
        -------
        norm : array-like, shape=[..., *base_shape]
            Point-wise norms.
        """
        sq_norm = self.pointwise_inner_product(
            tangent_vec_a=tangent_vec, tangent_vec_b=tangent_vec, base_point=base_point
        )
        return gs.sqrt(sq_norm)

    def exp(self, tangent_vec, base_point):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n_copies, *base_shape]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., n_copies, *base_shape]
            Point on the manifold.
            Optional, default: None.

        Returns
        -------
        exp : array-like, shape=[..., n_copies, *base_shape]
            Point on the manifold equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        base_manifold = self._space.base_manifold
        batch_shape = get_batch_shape(self._space.point_ndim, tangent_vec, base_point)

        tangent_vec, point_ = gs.broadcast_arrays(tangent_vec, base_point)
        point_ = gs.reshape(point_, (-1, *base_manifold.shape))
        vector_ = gs.reshape(tangent_vec, (-1, *base_manifold.shape))
        each_exp = base_manifold.metric.exp(vector_, point_)
        return gs.reshape(each_exp, batch_shape + self._space.shape)

    def log(self, point, base_point):
        """Compute the Riemannian logarithm of a point.

        Parameters
        ----------
        point : array-like, shape=[..., n_copies, *base_shape]
            Point on the manifold.
        base_point : array-like, shape=[..., n_copies, *base_shape]
            Point on the manifold.
            Optional, default: None.

        Returns
        -------
        log : array-like, shape=[..., n_copies, *base_shape]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        base_manifold = self._space.base_manifold
        batch_shape = get_batch_shape(self._space.point_ndim, point, base_point)

        point_, base_point_ = gs.broadcast_arrays(point, base_point)
        base_point_ = gs.reshape(base_point_, (-1, *base_manifold.shape))
        point_ = gs.reshape(point_, (-1, *base_manifold.shape))
        each_log = base_manifold.metric.log(point_, base_point_)
        return gs.reshape(each_log, batch_shape + self._space.shape)
