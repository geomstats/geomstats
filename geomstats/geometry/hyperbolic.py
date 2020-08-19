"""The n-dimensional hyperbolic space.

The n-dimensional hyperbolic space embedded and its different representations.
"""


import geomstats.backend as gs
import geomstats.vectorization
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric

TOLERANCE = 1e-6


class Hyperbolic(Manifold):
    """Class for the n-dimensional hyperbolic space.

    Class for the n-dimensional hyperbolic space
    as embedded in (n+1)-dimensional Minkowski space.

    The point_type variable allows to choose the
    representation of the points as input.

    If point_type is set to 'ball' then points are parametrized
    by their coordinates inside the Poincare Ball n-coordinates.

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

    default_coords_type = 'extrinsic'
    default_point_type = 'vector'

    def __init__(self, dim, scale=1):
        super(Hyperbolic, self).__init__(dim=dim)
        self.point_type = Hyperbolic.default_point_type
        self.coords_type = Hyperbolic.default_coords_type
        self.scale = scale
        self.metric = HyperbolicMetric(self.dim, self.scale)

    @staticmethod
    def _extrinsic_to_extrinsic_coordinates(point):
        """Convert the parameterization of a point.

        Convert the parameterization of a point in hyperbolic space
        from its extrinsic coordinates, to its intrinsic coordinates
        in Minkowski space.

        Parameters
        ----------
        point_extrinsic : array-like, shape=[..., dim + 1]
            Point in hyperbolic space in extrinsic coordinates.

        Returns
        -------
        point_intrinsic : array-like, shape=[..., dim]
            Point in hyperbolic space in intrinsic coordinates.
        """
        return point

    @staticmethod
    @geomstats.vectorization.decorator(['vector'])
    def _intrinsic_to_extrinsic_coordinates(point_intrinsic):
        """Convert intrinsic to extrinsic coordinates.

        Convert the parameterization of a point in hyperbolic space
        from its intrinsic coordinates, to its extrinsic coordinates
        in Minkowski space.

        Parameters
        ----------
        point_intrinsic : array-like, shape=[..., dim]
            Point in hyperbolic space in intrinsic coordinates.

        Returns
        -------
        point_extrinsic : array-like, shape=[..., dim + 1]
            Point in hyperbolic space in extrinsic coordinates.
        """
        coord_0 = gs.sqrt(1. + gs.linalg.norm(point_intrinsic, axis=-1) ** 2)
        coord_0 = gs.to_ndarray(coord_0, to_ndim=1)
        coord_0 = gs.to_ndarray(coord_0, to_ndim=2, axis=1)

        point_extrinsic = gs.hstack([coord_0, point_intrinsic])

        return point_extrinsic

    @staticmethod
    def _extrinsic_to_intrinsic_coordinates(point_extrinsic):
        """Convert extrinsic to intrinsic coordinates.

        Convert the parameterization of a point in hyperbolic space
        from its extrinsic coordinates in Minkowski space, to its
        intrinsic coordinates.

        Parameters
        ----------
        point_extrinsic : array-like, shape=[..., dim + 1]
            Point in hyperbolic space in extrinsic coordinates.

        Returns
        -------
        point_intrinsic : array-like, shape=[..., dim]
        """
        point_intrinsic = point_extrinsic[..., 1:]

        return point_intrinsic

    @staticmethod
    def _extrinsic_to_ball_coordinates(point):
        """Convert extrinsic to ball coordinates.

        Convert the parameterization of a point in hyperbolic space
        from its intrinsic coordinates, to the poincare ball model
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
        return point[..., 1:] / (1 + point[..., :1])

    @staticmethod
    @geomstats.vectorization.decorator(['vector'])
    def _ball_to_extrinsic_coordinates(point):
        """Convert ball to extrinsic coordinates.

        Convert the parameterization of a point in hyperbolic space
        from its poincare ball model coordinates, to the extrinsic
        coordinates.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in hyperbolic space in Poincare ball coordinates.

        Returns
        -------
        extrinsic : array-like, shape=[..., dim + 1]
            Point in hyperbolic space in extrinsic coordinates.
        """
        squared_norm = gs.sum(point**2, -1)
        denominator = 1 - squared_norm
        t = gs.to_ndarray((1 + squared_norm) / denominator, to_ndim=2, axis=1)
        expanded_denominator = gs.expand_dims(denominator, -1)
        expanded_denominator = gs.tile(
            expanded_denominator, (1, point.shape[-1]))
        intrinsic = (2 * point) / expanded_denominator
        return gs.concatenate([t, intrinsic], -1)

    @staticmethod
    @geomstats.vectorization.decorator(['vector'])
    def _half_plane_to_extrinsic_coordinates(point):
        """Convert half plane to extrinsic coordinates.

        Convert the parameterization of a point in the hyperbolic plane
        from its upper half plane model coordinates, to the extrinsic
        coordinates.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point in hyperbolic space in half-plane coordinates.

        Returns
        -------
        extrinsic : array-like, shape=[..., 3]
            Point in hyperbolic plane in extrinsic coordinates.
        """
        x, y = point[:, 0], point[:, 1]
        x2 = point[:, 0]**2
        den = x2 + (1 + y)**2
        x = gs.to_ndarray(x, to_ndim=2, axis=0)
        y = gs.to_ndarray(y, to_ndim=2, axis=0)
        x2 = gs.to_ndarray(x2, to_ndim=2, axis=0)
        den = gs.to_ndarray(den, to_ndim=2, axis=0)
        ball_point = gs.hstack((2 * x / den, (x2 + y**2 - 1) / den))
        return Hyperbolic._ball_to_extrinsic_coordinates(ball_point)

    @staticmethod
    @geomstats.vectorization.decorator(['vector'])
    def _extrinsic_to_half_plane_coordinates(point):
        """Convert extrinsic to half plane coordinates.

        Convert the parameterization of a point in the hyperbolic plane
        from its intrinsic coordinates, to the poincare upper half plane
        coordinates.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point in the hyperbolic plane in intrinsic coordinates.

        Returns
        -------
        point_half_plane : array-like, shape=[..., 2]
            Point in the hyperbolic plane in Poincare upper half-plane
            coordinates.
        """
        point_ball = \
            Hyperbolic._extrinsic_to_ball_coordinates(point)
        point_ball_x, point_ball_y = point_ball[:, 0], point_ball[:, 1]
        point_ball_x2 = point_ball_x**2
        denom = point_ball_x2 + (1 - point_ball_y)**2

        point_ball_x = gs.to_ndarray(
            point_ball_x, to_ndim=2, axis=0)
        point_ball_y = gs.to_ndarray(
            point_ball_y, to_ndim=2, axis=0)
        point_ball_x2 = gs.to_ndarray(
            point_ball_x2, to_ndim=2, axis=0)
        denom = gs.to_ndarray(
            denom, to_ndim=2, axis=0)

        point_half_plane = gs.hstack((
            (2 * point_ball_x) / denom,
            (1 - point_ball_x2 - point_ball_y**2) / denom))
        return point_half_plane

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
                Hyperbolic._ball_to_extrinsic_coordinates,
            'extrinsic-ball':
                Hyperbolic._extrinsic_to_ball_coordinates,
            'intrinsic-extrinsic':
                Hyperbolic._intrinsic_to_extrinsic_coordinates,
            'extrinsic-intrinsic':
                Hyperbolic._extrinsic_to_intrinsic_coordinates,
            'extrinsic-half-plane':
                Hyperbolic._extrinsic_to_half_plane_coordinates,
            'half-plane-extrinsic':
                Hyperbolic._half_plane_to_extrinsic_coordinates,
            'extrinsic-extrinsic':
                Hyperbolic._extrinsic_to_extrinsic_coordinates
        }

        if from_coordinates_system == to_coordinates_system:
            return point

        extrinsic =\
            coords_transform[from_coordinates_system +
                             '-extrinsic'](point)
        return \
            coords_transform['extrinsic-' +
                             to_coordinates_system](extrinsic)

    def belongs(self, point, tolerance=TOLERANCE):
        """Test if a point belongs to the hyperbolic space.

        Test if a point belongs to the hyperbolic space in
        its hyperboloid representation.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to be tested.
        tolerance : float
            Tolerance at which to evaluate how close the squared norm
            is to the reference value.
            Optional, default: TOLERANCE.

        Returns
        -------
        belongs : array-like, shape=[..., 1]
            Array of booleans indicating whether the corresponding points
            belong to the hyperbolic space.
        """
        raise NotImplementedError

    def to_coordinates(self, point, to_coords_type='ball'):
        """Convert coordinates of a point.

        Convert the parameterization of a point in the hyperbolic space
        from current coordinate system to the coordinate system given.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, dim + 1}]
            Point in hyperbolic space.
        to_point_type : str, {'extrinsic', 'intrinsic', etc}
            Coordinates type.
            Optional, default: 'ball'.

        Returns
        -------
        point_to : array-like, shape=[..., {dim, dim + 1}]
            Point in hyperbolic space in coordinates given by to_point_type.
        """
        return Hyperbolic.change_coordinates_system(point,
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
        from_point_type : str, {'ball', 'extrinsic', 'intrinsic', 'half_plane'}
            Coordinates type.

        Returns
        -------
        point_current : array-like, shape=[..., {dim, dim + 1}]
            Point in hyperbolic space.
        """
        return Hyperbolic.change_coordinates_system(
            point, from_coords_type, self.coords_type)

    def random_uniform(self, n_samples=1, bound=1.):
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

        samples = Hyperbolic.change_coordinates_system(
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
            signature=(dim, 0, 0))
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
