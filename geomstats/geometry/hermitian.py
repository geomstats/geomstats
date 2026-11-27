"""Complex Hermitian space.

Lead author: Yann Cabanes.
"""

import math

import geomstats.backend as gs
from geomstats.geometry.base import VectorSpace
from geomstats.geometry.diffeo import LinearIsomorphism
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.vectorization import repeat_out


class RealCoordinatesDiffeo(LinearIsomorphism):
    r"""Real coordinates of a complex vector.

    .. math::

        F(x + iy) = (x, y)

    It is a real linear isomorphism from :math:`\mathbb{C}^n` to
    :math:`\mathbb{R}^{2n}`, with the real parts first.

    Parameters
    ----------
    dim : int
        Complex dimension.
    """

    def __init__(self, dim):
        super().__init__(space_ndim=1)
        self.dim = dim

    def __call__(self, base_point):
        """Split a point into its real and imaginary parts.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        image_point : array-like, shape=[..., 2 * dim]
            Image point.
        """
        return gs.concatenate([gs.real(base_point), gs.imag(base_point)], axis=-1)

    def inverse(self, image_point):
        """Combine real and imaginary parts into a point.

        Parameters
        ----------
        image_point : array-like, shape=[..., 2 * dim]
            Image point.

        Returns
        -------
        base_point : array-like, shape=[..., dim]
            Base point.
        """
        return image_point[..., : self.dim] + 1j * image_point[..., self.dim :]


class Hermitian(VectorSpace):
    """Class for Hermitian spaces.

    By definition, a Hermitian space is a complex vector space
    of a given dimension, equipped with a Hermitian metric.

    Parameters
    ----------
    dim : int
        Complex dimension of the Hermitian space.
    equip : bool
        Whether to equip space with default metric.
    """

    def __init__(self, dim, equip=True):
        super().__init__(shape=(dim,), equip=equip, field="complex")

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return HermitianMetric

    @property
    def identity(self):
        """Identity of the group.

        Returns
        -------
        identity : array-like, shape=[n]
        """
        return gs.zeros(self.dim, dtype=gs.get_default_cdtype())

    def _create_basis(self):
        """Create the canonical basis."""
        return gs.eye(self.dim, dtype=gs.get_default_cdtype())

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
        return tangent_vec + base_point


class HermitianMetric(RiemannianMetric):
    """Class for the Hermitian metric.

    The inner product is the real part of the Hermitian product, and it is
    independent of the base point. The metric is flat, and geodesics are
    straight lines.

    Parameters
    ----------
    space : Hermitian
        Manifold.
    """

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
        mat = gs.eye(self._space.dim, dtype=gs.get_default_cdtype())
        if base_point is not None and base_point.ndim > 1:
            return gs.broadcast_to(mat, base_point.shape + (self._space.dim,))
        return mat

    @staticmethod
    def inner_product(tangent_vec_a, tangent_vec_b, base_point=None):
        """Inner product between two tangent vectors at a base point.

        It is the real part of the Hermitian product.

        Parameters
        ----------
        tangent_vec_a: array-like, shape=[..., dim]
            Tangent vector at base point.
        tangent_vec_b: array-like, shape=[..., dim]
            Tangent vector at base point.
        base_point: array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """
        return gs.real(gs.vecdot(tangent_vec_a, tangent_vec_b))

    @staticmethod
    def norm(vector, base_point=None):
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
        return gs.linalg.vector_norm(vector, axis=-1)

    @staticmethod
    def exp(tangent_vec, base_point):
        """Compute exp map of a base point in tangent vector direction.

        The Riemannian exponential is vector addition in the Hermitian space.

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
        return base_point + tangent_vec

    @staticmethod
    def log(point, base_point):
        """Compute log map using a base point and other point.

        The Riemannian logarithm is the subtraction in the Hermitian space.

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
        return point - base_point

    def parallel_transport(
        self, tangent_vec, base_point=None, direction=None, end_point=None
    ):
        r"""Compute the parallel transport of a tangent vector.

        On a Hermitian space, the parallel transport of a (tangent) vector
        returns the vector itself.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., dim]
            Point on the manifold. Point to transport from.
        direction : array-like, shape=[..., dim]
            Tangent vector at base point, along which the parallel transport
            is computed.
        end_point : array-like, shape=[..., dim]
            Point on the manifold. Point to transport to.

        Returns
        -------
        transported_tangent_vec : array-like, shape=[..., dim]
            Transported tangent vector at `exp_(base_point)(direction)`.
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

        In a Hermitian space, it is infinite everywhere.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        radius = gs.array(math.inf)
        return repeat_out(self._space.point_ndim, radius, base_point)
