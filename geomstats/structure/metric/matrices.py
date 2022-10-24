"""Module exposing the `Matrices` and `MatricesMetric` class."""

import geomstats.backend as gs
from geomstats.structure.geometry.metric import EuclideanMetric


class MatricesMetric(EuclideanMetric):
    """Euclidean metric on matrices given by Frobenius inner-product.
    """

    def __init__(self, space):
        super().__init__()
        self._space = space

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute Frobenius inner-product of two tangent vectors.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., m, n]
            Tangent vector.
        tangent_vec_b : array-like, shape=[..., m, n]
            Tangent vector.
        base_point : array-like, shape=[..., m, n]
            Base point.
            Optional, default: None.

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Frobenius inner-product of tangent_vec_a and tangent_vec_b.
        """
        return Matrices.frobenius_product(tangent_vec_a, tangent_vec_b)

    @staticmethod
    def norm(vector, base_point=None):
        """Compute norm of a matrix.

        Norm of a matrix associated to the Frobenius inner product.

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
        return gs.linalg.norm(vector, axis=(-2, -1))
