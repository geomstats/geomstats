"""The n-dimensional hyperbolic space.

Poincare half-space representation.

Lead author: Alice Le Brigant.
"""

import math

import geomstats.backend as gs
from geomstats.geometry._hyperbolic import HyperbolicDiffeo, _Hyperbolic
from geomstats.geometry.base import VectorSpaceOpenSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.vectorization import repeat_out


class PoincareHalfSpace(_Hyperbolic, VectorSpaceOpenSet):
    """Class for the n-dimensional Poincare half-space.

    Class for the n-dimensional Poincaré half space model. For other
    representations of hyperbolic spaces see the `Hyperbolic` class.

    Parameters
    ----------
    dim : int
        Dimension of the hyperbolic space.
    """

    def __init__(self, dim, equip=True):
        self.coords_type = "half-space"
        super().__init__(
            dim=dim,
            embedding_space=Euclidean(dim),
            intrinsic=True,
            equip=equip,
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return PoincareHalfSpaceMetric

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the upper half space.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to be checked.
        atol : float
            Absolute tolerance to evaluate positivity of the last coordinate.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Array of booleans indicating whether the corresponding
            points belong to the upper half space.
        """
        point_dim = point.shape[-1]
        belongs = point_dim == self.dim
        return gs.logical_and(belongs, point[..., -1] >= -atol)

    def projection(self, point, atol=gs.atol):
        """Project a point in ambient space to the open set.

        The last coordinate is floored to `atol` if it is negative.

        Parameters
        ----------
        point : array-like, shape=[..., dim_embedding]
            Point in ambient space.
        atol : float
            Tolerance to evaluate positivity.

        Returns
        -------
        projected : array-like, shape=[..., dim_embedding]
            Projected point.
        """
        last = gs.where(point[..., -1] < atol, atol, point[..., -1])
        return gs.concatenate([point[..., :-1], last[..., None]], axis=-1)


class PoincareHalfSpaceMetric(PullbackDiffeoMetric):
    """Class for the metric of the n-dimensional hyperbolic space.

    Class for the metric of the n-dimensional hyperbolic space
    as embedded in the Poincaré half space model.
    """

    def __init__(self, space):
        image_space = PoincareBall(dim=space.dim)
        super().__init__(
            space=space,
            image_space=image_space,
            diffeo=HyperbolicDiffeo(space.coords_type, image_space.coords_type),
        )

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the inner-product of two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim + 1]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., dim + 1]
            Second tangent vector at base point.
        base_point : array-like, shape=[..., dim + 1]
            Point in hyperbolic space.

        Returns
        -------
        inner_prod : array-like, shape=[..., 1]
            Inner-product of the two tangent vectors.
        """
        inner_prod = gs.sum(tangent_vec_a * tangent_vec_b, axis=-1)
        return inner_prod / base_point[..., -1] ** 2

    def injectivity_radius(self, base_point=None):
        """Compute the radius of the injectivity domain.

        This is is the supremum of radii r for which the exponential map is a
        diffeomorphism from the open ball of radius r centered at the base
        point onto its image.
        In the case of the hyperbolic space, it does not depend on the base
        point and is infinite everywhere, because of the negative curvature.

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
