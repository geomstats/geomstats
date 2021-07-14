"""Base class for the n-dimensional hyperbolic space.

The n-dimensional hyperbolic regardless its different representations.
"""


import geomstats.backend as gs
from geomstats.geometry.riemannian_metric import RiemannianMetric


class _Hyperbolic:
    """Class for the n-dimensional hyperbolic space.

    Common class for the n-dimensional hyperbolic space regardless the
    representation being used. This class gathers the change of coordinate
    methods that are common to all representations.

    This class cannot be instantiated in itself, but a public class
    `Hyperbolic` is available, that returns the class that corresponds to the
    chosen representation: Hyperboloid, Poincare Ball or Poincare Half-space.

    Parameters
    ----------
    dim : int
        Dimension of the hyperbolic space.
    point_type : str, {'extrinsic', 'intrinsic', etc}
        Default coordinates to represent points in hyperbolic space.
        Optional, default: 'extrinsic'.
    scale : int
        Scale of the hyperbolic space, defined as the set of points
        in Minkowski space whose squared norm is equal to -scale.
        Optional, default: 1.
    """

    def __init__(self, dim, scale=1, **kwargs):
        super(_Hyperbolic, self).__init__(dim=dim, **kwargs)
        self.dim = dim
        self.scale = scale

    @staticmethod
    def _extrinsic_to_extrinsic_coordinates(point):
        """Convert the parameterization of a point.

        Dummy method that returns the input point.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Point in hyperbolic space in extrinsic coordinates.

        Returns
        -------
        point_extrinsic : array-like, shape=[..., dim]
            Point in hyperbolic space in extrinsic coordinates.
        """
        return point

    @staticmethod
    def _intrinsic_to_extrinsic_coordinates(point):
        """Convert intrinsic to extrinsic coordinates.

        Convert the parameterization of a point in hyperbolic space
        from its intrinsic coordinates, to its extrinsic coordinates
        in Minkowski space.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in hyperbolic space in intrinsic coordinates.

        Returns
        -------
        point_extrinsic : array-like, shape=[..., dim + 1]
            Point in hyperbolic space in extrinsic coordinates.
        """
        coord_0 = gs.sqrt(1. + gs.sum(point ** 2, axis=-1))
        point_extrinsic = gs.concatenate(
            [coord_0[..., None], point], axis=-1)

        return point_extrinsic

    @staticmethod
    def _extrinsic_to_intrinsic_coordinates(point):
        """Convert extrinsic to intrinsic coordinates.

        Convert the parameterization of a point in hyperbolic space
        from its extrinsic coordinates in Minkowski space, to its
        intrinsic coordinates.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Point in hyperbolic space in extrinsic coordinates.

        Returns
        -------
        point_intrinsic : array-like, shape=[..., dim]
        """
        point_intrinsic = point[..., 1:]

        return point_intrinsic

    @staticmethod
    def _extrinsic_to_ball_coordinates(point):
        """Convert extrinsic to ball coordinates.

        Convert the parameterization of a point in hyperbolic space
        from its intrinsic coordinates, to the Poincare ball model
        coordinates.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Point in hyperbolic space in extrinsic coordinates.

        Returns
        -------
        point_ball : array-like, shape=[..., dim]
            Point in hyperbolic space in Poincare ball coordinates.
        """
        point_ball = point[..., 1:] / (1 + point[..., :1])

        return point_ball

    @staticmethod
    def _ball_to_extrinsic_coordinates(point):
        """Convert ball to extrinsic coordinates.

        Convert the parameterization of a point in hyperbolic space
        from its Poincare ball model coordinates, to the extrinsic
        coordinates.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in hyperbolic space in Poincare ball coordinates.

        Returns
        -------
        point_extrinsic : array-like, shape=[..., dim + 1]
            Point in hyperbolic space in extrinsic coordinates.
        """
        squared_norm = gs.sum(point**2, -1)
        denominator = 1 - squared_norm
        t = (1 + squared_norm) / denominator
        intrinsic = gs.einsum(
            '...i, ...->...i', 2 * point, 1. / denominator)
        point_extrinsic = gs.concatenate([t[..., None], intrinsic], -1)
        return point_extrinsic

    @classmethod
    def _half_space_to_extrinsic_coordinates(cls, point):
        """Convert half-space to extrinsic coordinates.

        Convert the parameterization of a point in the hyperbolic space
        from its Poincare half-space coordinates, to the extrinsic
        coordinates.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in hyperbolic space in half-space coordinates.

        Returns
        -------
        point_extrinsic : array-like, shape=[..., dim + 1]
            Point in hyperbolic space in extrinsic coordinates.
        """
        point_ball = cls.half_space_to_ball_coordinates(point)
        point_extrinsic = cls._ball_to_extrinsic_coordinates(point_ball)

        return point_extrinsic

    @classmethod
    def _extrinsic_to_half_space_coordinates(cls, point):
        """Convert extrinsic to half-space coordinates.

        Convert the parameterization of a point in the hyperbolic space
        from its extrinsic coordinates, to the Poincare half-space
        coordinates.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Point in the hyperbolic space in extrinsic coordinates.

        Returns
        -------
        point_half_space : array-like, shape=[..., dim]
            Point in the hyperbolic space in half-space coordinates.
        """
        point_ball = \
            cls._extrinsic_to_ball_coordinates(point)
        point_half_space = cls.ball_to_half_space_coordinates(point_ball)

        return point_half_space

    @staticmethod
    def half_space_to_ball_coordinates(point):
        """Convert half-space to ball coordinates.

        Convert the parameterization of a point in the hyperbolic space
        from its Poincare half-space coordinates, to the Poincare ball
        coordinates.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in the hyperbolic space in half-space coordinates.

        Returns
        -------
        point_ball : array-like, shape=[..., dim]
            Point in the hyperbolic space in Poincare ball coordinates.
        """
        sq_norm = gs.sum(point ** 2, axis=-1)
        den = 1 + sq_norm + 2 * point[..., -1]
        component_1 = gs.einsum(
            '...i,...->...i', point[..., :-1], 2. / den)
        component_2 = (sq_norm - 1) / den
        point_ball = gs.concatenate(
            [component_1, component_2[..., None]], axis=-1)

        return point_ball

    @staticmethod
    def ball_to_half_space_coordinates(point):
        """Convert ball to half space coordinates.

        Convert the parameterization of a point in the hyperbolic space
        from its Poincare ball coordinates, to the Poincare half-space
        coordinates.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in the hyperbolic space in Poincare ball coordinates.

        Returns
        -------
        point_ball : array-like, shape=[..., dim]
            Point in the hyperbolic space in half-space coordinates.
        """
        sq_norm = gs.sum(point ** 2, axis=-1)
        den = 1 + sq_norm - 2 * point[..., -1]
        component_1 = gs.einsum(
            '...i,...->...i', point[..., :-1], 2. / den)
        component_2 = (1 - sq_norm) / den
        point_half_space = gs.concatenate(
            [component_1, component_2[..., None]], axis=-1)

        return point_half_space

    @staticmethod
    def half_space_to_ball_tangent(tangent_vec, base_point):
        """Convert half-space to ball tangent coordinates.

        Convert the parameterization of a tangent vector to the
        hyperbolic space from its Poincare half-space coordinates, to
        the Poinare ball coordinates.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at the base point in the Poincare half-space.
        base_point : array-like, shape=[..., dim]
            Point in the Poincare half-space.

        Returns
        -------
        tangent_vec_ball : array-like, shape=[..., dim]
           Tangent vector in the Poincare ball.
        """
        sq_norm = gs.sum(base_point ** 2, axis=-1)
        den = 1. + sq_norm + 2. * base_point[..., -1]
        scalar_prod = gs.sum(base_point * tangent_vec, axis=-1)
        component_1 = (
            gs.einsum('...i,...->...i', tangent_vec[..., :-1], 2. / den)
            - 4. * gs.einsum(
                '...i,...->...i', base_point[..., :-1],
                (scalar_prod + tangent_vec[..., -1]) / den**2))
        component_2 = 2 * scalar_prod / den \
            - 2 * (sq_norm - 1) * (scalar_prod + tangent_vec[..., -1]) \
            / den ** 2
        tangent_vec_ball = gs.concatenate(
            [component_1, component_2[..., None]], axis=-1)

        return tangent_vec_ball

    @staticmethod
    def ball_to_half_space_tangent(tangent_vec, base_point):
        """Convert ball to half-space tangent coordinates.

        Convert the parameterization of a tangent vector to the
        hyperbolic space from its Poincare ball coordinates, to
        the Poinare half-space coordinates.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at the base point in the Poincare ball.
        base_point : array-like, shape=[..., dim]
            Point in the Poincare ball.

        Returns
        -------
        tangent_vec_half_spacel : array-like, shape=[..., dim]
            Tangent vector in the Poincare half-space.

        """
        sq_norm = gs.sum(base_point ** 2, axis=-1)
        den = 1 + sq_norm - 2 * base_point[..., -1]
        scalar_prod = gs.sum(base_point * tangent_vec, -1)
        component_1 = (
            gs.einsum('...i,...->...i', tangent_vec[..., :-1], 2. / den)
            - 4. * gs.einsum(
                '...i,...->...i', base_point[..., :-1],
                (scalar_prod - tangent_vec[..., -1]) / den**2))
        component_2 = -2. * scalar_prod / den \
            - 2 * (1. - sq_norm) * (scalar_prod - tangent_vec[..., -1]) \
            / den**2
        tangent_vec_half_space = gs.concatenate(
            [component_1, component_2[..., None]], axis=-1)

        return tangent_vec_half_space

    @staticmethod
    def change_coordinates_system(point,
                                  from_coordinates_system,
                                  to_coordinates_system):
        """Convert coordinates of a point.

        Convert the parameterization of a point in the hyperbolic space
        from current given coordinate system to an other also given in
        parameters. The possible coordinates system are 'extrinsic',
        'intrinsic', 'ball' and 'half-plane' that correspond respectivelly
        to extrinsic coordinates in the hyperboloid , intrinsic
        coordinates in the hyperboloid, ball coordinates in the Poincare
        ball model and coordinates in the Poincare upper half-plane model.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, dim + 1}]
            Point in hyperbolic space.
        from_coordinates_system : str, {'extrinsic', 'intrinsic', etc}
            Coordinates type.
        to_coordinates_system : str, {'extrinsic', 'intrinsic', etc}
            Coordinates type.

        Returns
        -------
        point_to : array-like, shape=[..., dim]
                               or shape=[n_sample, dim + 1]
            Point in hyperbolic space in coordinates given by to_point_type.
        """
        coords_transform = {
            'ball-extrinsic':
                _Hyperbolic._ball_to_extrinsic_coordinates,
            'extrinsic-ball':
                _Hyperbolic._extrinsic_to_ball_coordinates,
            'intrinsic-extrinsic':
                _Hyperbolic._intrinsic_to_extrinsic_coordinates,
            'extrinsic-intrinsic':
                _Hyperbolic._extrinsic_to_intrinsic_coordinates,
            'extrinsic-half-space':
                _Hyperbolic._extrinsic_to_half_space_coordinates,
            'half-space-extrinsic':
                _Hyperbolic._half_space_to_extrinsic_coordinates,
            'extrinsic-extrinsic':
                _Hyperbolic._extrinsic_to_extrinsic_coordinates
        }

        if from_coordinates_system == to_coordinates_system:
            return point

        extrinsic =\
            coords_transform[from_coordinates_system +
                             '-extrinsic'](point)
        return \
            coords_transform['extrinsic-' +
                             to_coordinates_system](extrinsic)

    def to_coordinates(self, point, to_coords_type='ball'):
        """Convert coordinates of a point.

        Convert the parameterization of a point in the hyperbolic space
        from current coordinate system to the coordinate system given.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, dim + 1}]
            Point in hyperbolic space.
        to_coords_type : str, {'extrinsic', 'intrinsic', etc}
            Coordinates type.
            Optional, default: 'ball'.

        Returns
        -------
        point_to : array-like, shape=[..., {dim, dim + 1}]
            Point in hyperbolic space in coordinates given by to_point_type.
        """
        return _Hyperbolic.change_coordinates_system(point,
                                                     self.coords_type,
                                                     to_coords_type)

    def from_coordinates(self, point, from_coords_type):
        """Convert to a type of coordinates given some type.

        Convert the parameterization of a point in hyperbolic space
        from given coordinate system to the current coordinate system.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, dim + 1}]
            Point in hyperbolic space in coordinates from_point_type.
        from_coords_type : str, {'ball', 'extrinsic', 'intrinsic', ...}
            Coordinates type.

        Returns
        -------
        point_current : array-like, shape=[..., {dim, dim + 1}]
            Point in hyperbolic space.
        """
        return _Hyperbolic.change_coordinates_system(
            point, from_coords_type, self.coords_type)

    def random_point(self, n_samples=1, bound=1.):
        """Sample over the hyperbolic space using uniform distribution.

        Sample over the hyperbolic space. The sampling is performed
        by sampling over uniform distribution, the sampled examples
        are considered in the intrinsic coordinates system.
        The function then transforms intrinsic samples into system
        coordinate selected.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound: float
            Bound defining the hypersquare in which to sample uniformly.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., dim + 1]
            Samples in hyperbolic space.
        """
        size = (n_samples, self.dim)
        samples = bound * 2. * (gs.random.rand(*size) - 0.5)

        samples = _Hyperbolic.change_coordinates_system(
            samples, 'intrinsic', self.coords_type)

        if n_samples == 1:
            samples = gs.squeeze(samples, axis=0)
        return samples


class HyperbolicMetric(RiemannianMetric):
    """Class that defines operations using a hyperbolic metric.

    Parameters
    ----------
    dim : int
        Dimension of the hyperbolic space.
    point_type : str, {'extrinsic', 'intrinsic', etc}, optional
        Default coordinates to represent points in hyperbolic space.
    scale : int, optional
        Scale of the hyperbolic space, defined as the set of points
        in Minkowski space whose squared norm is equal to -scale.
    """

    default_point_type = 'vector'
    default_coords_type = 'extrinsic'

    def __init__(self, dim, scale=1):
        super(HyperbolicMetric, self).__init__(
            dim=dim,
            signature=(dim, 0))
        self.point_type = HyperbolicMetric.default_point_type

        self.scale = scale

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the inner-product of two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim + 1]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., dim + 1]
            Second tangent vector at base point.
        base_point : array-like, shape=[..., dim + 1]
            Point in hyperbolic space.
            Optional, default: None.

        Returns
        -------
        inner_prod : array-like, shape=[..., 1]
            Inner-product of the two tangent vectors.
        """
        inner_prod = self._inner_product(
            tangent_vec_a, tangent_vec_b, base_point)
        inner_prod *= self.scale ** 2
        return inner_prod

    def squared_norm(self, vector, base_point=None):
        """Compute the squared norm of a vector.

        Squared norm of a vector associated with the inner-product
        at the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim + 1]
            Vector on the tangent space of the hyperbolic space at base point.
        base_point : array-like, shape=[..., dim + 1]
            Point in hyperbolic space in extrinsic coordinates.
            Optional, default: None.

        Returns
        -------
        sq_norm : array-like, shape=[..., 1]
            Squared norm of the vector.
        """
        sq_norm = self._squared_norm(vector)
        sq_norm *= self.scale ** 2
        return sq_norm

    def _squared_norm(self, vector, base_point=None):
        """Squared norm with hyperbolic scale 1.

        Squared norm of a vector associated with the inner-product
        at the tangent space at a base point with scale=1.

        Parameters
        ----------
        vector : array-like, shape=[..., dim + 1]
            Vector on the tangent space of the hyperbolic space at base point.
        base_point : array-like, shape=[..., dim + 1]
            Point in hyperbolic space in extrinsic coordinates.
            Optional, default: None.

        Returns
        -------
        sq_norm : array-like, shape=[..., 1]
            Squared norm of the vector.
        """
        return super().squared_norm(vector, base_point=base_point)

    def _inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the inner-product of two tangent vectors with scale=1.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim + 1]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., dim + 1]
            Second tangent vector at base point.
        base_point : array-like, shape=[..., dim + 1]
            Point in hyperbolic space.
            Optional, default: None.

        Returns
        -------
        inner_prod : array-like, shape=[..., 1]
            Inner-product of the two tangent vectors.
        """
        return super().inner_product(tangent_vec_a, tangent_vec_b,
                                     base_point=base_point)
