"""Complex Hermitian space.

Lead author: Yann Cabanes.
"""

import geomstats.backend as gs
from geomstats.geometry.base import ComplexVectorSpace
from geomstats.geometry.complex_riemannian_metric import ComplexRiemannianMetric


class Hermitian(ComplexVectorSpace):
    """Class for Hermitian spaces.

    By definition, a Hermitian space is a complex vector space
    of a given dimension, equipped with a Hermitian metric.

    Parameters
    ----------
    dim : int
        Dimension of the Hermitian space.
    """

    def __init__(self, dim, equip=True):
        super().__init__(shape=(dim,), equip=equip)

    def _default_metric(self):
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


class HermitianMetric(ComplexRiemannianMetric):
    """Class for Hermitian metrics.

    As a Riemannian metric, the Hermitian metric is:

    - flat: the inner-product is independent of the base point.
    - positive definite: it has signature (dimension, 0, 0),
      where dimension is the dimension of the Hermitian space.
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
        return gs.dot(gs.conj(tangent_vec_a), tangent_vec_b)

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
        return gs.linalg.norm(vector, axis=-1)

    @staticmethod
    def exp(tangent_vec, base_point, **kwargs):
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
    def log(point, base_point, **kwargs):
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
