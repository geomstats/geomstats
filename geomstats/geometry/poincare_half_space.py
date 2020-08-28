"""The n-dimensional hyperbolic space.

Poincare half-space representation.
"""

import geomstats.backend as gs
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.geometry.riemannian_metric import RiemannianMetric


class PoincareHalfSpace(Hyperbolic):
    """Class for the n-dimensional hyperbolic space.

    Class for the n-dimensional hyperbolic space
    as embedded in the Poincaré half space model.

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
            dim=dim,
            scale=scale)
        self.coords_type = PoincareHalfSpace.default_coords_type
        self.point_type = PoincareHalfSpace.default_point_type
        self.metric = PoincareHalfSpaceMetric(self.dim, self.scale)

    def belongs(self, point):
        """Test if a point belongs to the hyperbolic space.

        Test if a point belongs to the hyperbolic space based on
        the poincare ball representation.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to be tested.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Array of booleans indicating whether the corresponding points
            belong to the hyperbolic space.
        """
        return point[..., -1] > 0


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
            dim=dim,
            signature=(dim, 0, 0))
        self.coords_type = PoincareHalfSpace.default_coords_type
        self.point_type = PoincareHalfSpace.default_point_type
        self.scale = scale
        self.poincare_ball = PoincareBall(dim=dim, scale=scale)

    def exp(self, tangent_vec, base_point):
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
