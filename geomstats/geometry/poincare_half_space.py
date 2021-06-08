"""The n-dimensional hyperbolic space.

Poincare half-space representation.
"""

import geomstats.backend as gs
from geomstats.geometry._hyperbolic import _Hyperbolic
from geomstats.geometry.base import OpenSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.geometry.riemannian_metric import RiemannianMetric


class PoincareHalfSpace(_Hyperbolic, OpenSet):
    """Class for the n-dimensional Poincare half-space.

    Class for the n-dimensional Poincaré half space model. For other
    representations of hyperbolic spaces see the `Hyperbolic` class.

    Parameters
    ----------
    dim : int
        Dimension of the hyperbolic space.
    scale : int, optional
        Scale of the hyperbolic space, defined as the set of points
        in Minkowski space whose squared norm is equal to -scale.
    """

    default_coords_type = 'half-space'
    default_point_type = 'vector'

    def __init__(self, dim, scale=1):
        super(PoincareHalfSpace, self).__init__(
            dim=dim, ambient_space=Euclidean(dim), scale=scale,
            metric=PoincareHalfSpaceMetric(dim, scale))
        self.coords_type = PoincareHalfSpace.default_coords_type
        self.point_type = PoincareHalfSpace.default_point_type

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
        belongs = gs.logical_and(belongs, point[..., -1] >= atol)
        return belongs

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
        projected = gs.concatenate([point[..., :-1], last[..., None]], axis=-1)
        return projected


class PoincareHalfSpaceMetric(RiemannianMetric):
    """Class for the metric of the n-dimensional hyperbolic space.

    Class for the metric of the n-dimensional hyperbolic space
    as embedded in the Poincaré half space model.

    Parameters
    ----------
    dim : int
        Dimension of the hyperbolic space.
    scale : int
        Scale of the hyperbolic space, defined as the set of points
        in Minkowski space whose squared norm is equal to -scale.
        Optional, default: 1.
    """

    default_point_type = 'vector'
    default_coords_type = 'half-space'

    def __init__(self, dim, scale=1.):
        super(PoincareHalfSpaceMetric, self).__init__(
            dim=dim, signature=(dim, 0))
        self.coords_type = PoincareHalfSpace.default_coords_type
        self.point_type = PoincareHalfSpace.default_point_type
        self.scale = scale
        self.poincare_ball = PoincareBall(dim=dim, scale=scale)

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
        inner_prod = inner_prod / base_point[..., -1]**2
        return inner_prod

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the Riemannian exponential.

        Parameters
        ----------
        tangent_vec : array-like, shape=[...,n]
            Tangent vector at the base point in the Poincare half space.
        base_point : array-like, shape=[...,n]
            Point in the Poincare half space.

        Returns
        -------
        end_point : array-like, shape=[...,n]
            Point in the Poincare half space, reached by the geodesic
            starting from `base_point` with initial velocity `tangent_vec`
        """
        base_point_ball = self.poincare_ball.half_space_to_ball_coordinates(
            base_point)
        tangent_vec_ball = self.poincare_ball.half_space_to_ball_tangent(
            tangent_vec, base_point)
        end_point_ball = self.poincare_ball.metric.exp(
            tangent_vec_ball, base_point_ball)
        end_point = self.poincare_ball.ball_to_half_space_coordinates(
            end_point_ball)
        return end_point

    def log(self, point, base_point, **kwargs):
        """Compute Riemannian logarithm of a point wrt a base point.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in hyperbolic space.
        base_point : array-like, shape=[..., dim]
            Point in hyperbolic space.

        Returns
        -------
        log : array-like, shape=[..., dim]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        point_ball = self.poincare_ball.half_space_to_ball_coordinates(point)
        base_point_ball = self.poincare_ball.half_space_to_ball_coordinates(
            base_point)
        log_ball = self.poincare_ball.metric.log(point_ball, base_point_ball)
        log = self.poincare_ball.ball_to_half_space_tangent(
            log_ball, base_point_ball)
        return log
