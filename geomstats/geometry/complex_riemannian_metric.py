"""Riemannian and pseudo-Riemannian metrics for complex manifolds.

Lead author: Yann Cabanes.
"""

import geomstats.backend as gs
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.vectorization import check_is_batch


class ComplexRiemannianMetric(RiemannianMetric):
    r"""Class for Riemannian and pseudo-Riemannian metrics for Complex manifolds.

    The associated Levi-Civita connection on the tangent bundle.
    """

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
        inner_prod_mat = self.metric_matrix(base_point)
        aux = gs.einsum("...j,...jk->...k", gs.conj(tangent_vec_a), inner_prod_mat)
        return gs.einsum("...i,...i->...", aux, tangent_vec_b)

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

        Returns
        -------
        sq_norm : array-like, shape=[...,]
            Squared norm.
        """
        sq_norm = self.inner_product(vector, vector, base_point)
        return gs.real(sq_norm)

    def random_unit_tangent_vec(self, base_point=None, n_vectors=1):
        """Generate a random unit tangent vector at a given point.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point.
        n_vectors : float
            Number of vectors to be generated at base_point.
            For vectorization purposes n_vectors can be greater than 1 iff
            base_point consists of a single point.

        Returns
        -------
        normalized_vector : array-like, shape=[..., n_vectors, dim]
            Random unit tangent vector at base_point.
        """
        is_batch = check_is_batch(self._space.point_ndim, base_point)
        if is_batch and n_vectors > 1:
            raise ValueError(
                "Several tangent vectors is only applicable to a single base point."
            )
        dtype = gs.get_default_cdtype() if base_point is None else base_point.dtype
        point_shape = self._space.shape
        random_vector = gs.squeeze(
            gs.cast(gs.random.rand(n_vectors, *point_shape), dtype=dtype)
            + 1j * gs.cast(gs.random.rand(n_vectors, *point_shape), dtype=dtype)
        )
        normalized_vector = self.normalize(random_vector, base_point)
        return gs.squeeze(normalized_vector)
