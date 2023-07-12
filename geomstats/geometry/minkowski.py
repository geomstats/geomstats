"""Minkowski space.

Lead author: Nina Miolane.
"""

import math

import geomstats.backend as gs
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric


class Minkowski(Euclidean):
    """Class for Minkowski space.

    This is the Euclidean space endowed with the inner-product of signature (
    dim-1, 1).

    Parameters
    ----------
    dim : int
       Dimension of Minkowski space.
    """

    def __new__(cls, dim, equip=True):
        """Instantiate a Minkowski space.

        This is an instance of the `Euclidean` class endowed with the
        `MinkowskiMetric`.
        """
        space = Euclidean(dim, equip=False)
        if equip:
            space.equip_with_metric(MinkowskiMetric)
        return space


class MinkowskiMetric(RiemannianMetric):
    """Class for the pseudo-Riemannian Minkowski metric.

    The metric is flat: the inner product is independent of the base point.
    """

    def __init__(self, space):
        super().__init__(space=space, signature=(space.dim - 1, 1))

    def metric_matrix(self, base_point=None):
        """Compute the inner product matrix, independent of the base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        inner_prod_mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        q, p = self.signature
        diagonal = gs.array([-1.0] * p + [1.0] * q)
        mat = from_vector_to_diagonal_matrix(diagonal)
        if base_point is not None and base_point.ndim > 1:
            mat = gs.broadcast_to(mat, base_point.shape + (p + q,))
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
        q, p = self.signature
        diagonal = gs.array([-1.0] * p + [1.0] * q, dtype=tangent_vec_a.dtype)
        return gs.dot(diagonal * tangent_vec_a, tangent_vec_b)

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the Riemannian exponential of `tangent_vec` at `base_point`.

        The Riemannian exponential is the addition in the Minkowski space.

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
        """Compute the Riemannian logarithm of `point` at `base_point`.

        The Riemannian logarithm is the subtraction in the Minkowski space.

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
        log = point - base_point
        return log

    def injectivity_radius(self, base_point):
        """Compute the radius of the injectivity domain.

        This is is the supremum of radii r for which the exponential map is a
        diffeomorphism from the open ball of radius r centered at the base
        point onto its image.
        In the case of the Minkowski space, it does not depend on the base
        point and is infinite everywhere, because of the flat curvature.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        radius : float
            Injectivity radius.
        """
        return math.inf
