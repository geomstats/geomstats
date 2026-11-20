"""Euclidean space."""

import math

import geomstats.backend as gs
from geomstats.geometry.base import VectorSpace
from geomstats.geometry.diffeo import LinearIsomorphism
from geomstats.geometry.riemannian_metric import ConstantMetricMatrixRiemannianMetric
from geomstats.vectorization import repeat_out


class CoordinateScalingDiffeo(LinearIsomorphism):
    r"""Scaling of each coordinate by a nonzero factor.

    .. math::

        F(x) = (s_1 x_1, \dots, s_n x_n)

    Parameters
    ----------
    scales : array-like, shape=[dim]
        Nonzero scaling factors.
    """

    def __init__(self, scales):
        super().__init__(space_ndim=1)
        self.scales = scales

    def __call__(self, base_point):
        """Scale the coordinates of a point.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        image_point : array-like, shape=[..., dim]
            Image point.
        """
        return self.scales * base_point

    def inverse(self, image_point):
        """Unscale the coordinates of a point.

        Parameters
        ----------
        image_point : array-like, shape=[..., dim]
            Image point.

        Returns
        -------
        base_point : array-like, shape=[..., dim]
            Base point.
        """
        return image_point / self.scales


class Euclidean(VectorSpace):
    """Class for Euclidean spaces.

    By definition, a Euclidean space is a vector space of a given
    dimension, equipped with a Euclidean metric.

    Parameters
    ----------
    dim : int
        Dimension of the Euclidean space.
    """

    def __init__(self, dim, equip=True):
        super().__init__(
            dim=dim,
            shape=(dim,),
            equip=equip,
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return CanonicalEuclideanMetric

    @property
    def identity(self):
        """Identity of the group.

        Returns
        -------
        identity : array-like, shape=[n]
        """
        return gs.zeros(self.dim)

    def _create_basis(self):
        """Create the canonical basis."""
        return gs.eye(self.dim)

    def exp(self, tangent_vec, base_point):
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
        return tangent_vec + base_point


class EuclideanMetric(ConstantMetricMatrixRiemannianMetric):
    """Class for a Euclidean metric.

    Points are vectors, and the metric matrix is independent of the base
    point. The metric is flat, and geodesics are straight lines.

    Parameters
    ----------
    space : Manifold object
        M in the tuple (M, g).
    metric_matrix : array-like, shape=[dim, dim]
        Metric matrix.
        Optional, default: identity.
    signature : tuple
        Signature of the metric.
        Optional, default: (dim, 0).
    """

    def log(self, point, base_point):
        """Compute log map using a base point and other point.

        The Riemannian logarithm is the subtraction in the Euclidean space.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point.
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        log : array-like, shape=[..., dim]
            Riemannian logarithm.
        """
        return point - base_point

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
        transported_tangent_vec : array-like, shape=[..., dim]
            Transported tangent vector at `exp_(base_point)(tangent_vec_b)`.
        """
        transported_tangent_vec = gs.copy(tangent_vec)
        return repeat_out(
            self._space.point_ndim,
            transported_tangent_vec,
            tangent_vec,
            base_point,
            direction,
            end_point,
            out_shape=self._space.shape,
        )

    def injectivity_radius(self, base_point=None):
        """Compute the radius of the injectivity domain.

        This is is the supremum of radii r for which the exponential map is a
        diffeomorphism from the open ball of radius r centered at the base
        point onto its image.

        Parameters
        ----------
        base_point : array-like, shape=[..., {dim, [n, m]}]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        radius = gs.array(math.inf)
        return repeat_out(self._space.point_ndim, radius, base_point)


class CanonicalEuclideanMetric(EuclideanMetric):
    """Class for the canonical Euclidean metric.

    Notes
    -----
    Metric matrix is identity (NB: `EuclideanMetric` allows
    to use a different metric matrix).
    """

    def __init__(self, space):
        super().__init__(space)

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
        inner_product = gs.dot(tangent_vec_a, tangent_vec_b)
        return repeat_out(
            self._space.point_ndim,
            inner_product,
            tangent_vec_a,
            tangent_vec_b,
            base_point,
        )

    def squared_norm(self, vector, base_point=None):
        """Compute the square of the norm of a vector.

        Squared norm of a vector associated to the inner product
        at the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        sq_norm : array-like, shape=[...,]
            Squared norm.
        """
        sq_norm = gs.linalg.norm(vector, axis=-1) ** 2
        return repeat_out(self._space.point_ndim, sq_norm, vector, base_point)

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
        norm = gs.linalg.norm(vector, axis=-1)
        return repeat_out(self._space.point_ndim, norm, vector, base_point)

    def flat(self, tangent_vec, base_point=None):
        """Map a tangent vector to a covector.

        The metric matrix is the identity, so this is the identity.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        cotangent_vec : array-like, shape=[..., dim]
            Covector at base point.
        """
        return repeat_out(
            self._space.point_ndim,
            gs.copy(tangent_vec),
            tangent_vec,
            base_point,
            out_shape=self._space.shape,
        )

    def sharp(self, cotangent_vec, base_point=None):
        """Map a covector to a tangent vector.

        The metric matrix is the identity, so this is the identity.

        Parameters
        ----------
        cotangent_vec : array-like, shape=[..., dim]
            Covector at base point.
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        return repeat_out(
            self._space.point_ndim,
            gs.copy(cotangent_vec),
            cotangent_vec,
            base_point,
            out_shape=self._space.shape,
        )
