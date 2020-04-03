"""The n-dimensional hyperbolic space.

The n-dimensional hyperbolic space embedded and its different representations.
"""

import math


import geomstats.backend as gs
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.riemannian_metric import RiemannianMetric


class Hyperbolic(EmbeddedManifold):
    """Class for the n-dimensional hyperbolic space.

    Class for the n-dimensional hyperbolic space
    as embedded in (n+1)-dimensional Minkowski space.

    The point_type variable allows to choose the
    representation of the points as input.

    If point_type is set to 'ball' then points are parametrized
    by their coordinates inside the Poincare Ball n-coordinates.

    Parameters
    ----------
    dimension : int
        Dimension of the hyperbolic space.
    point_type : str, {'extrinsic', 'intrinsic', etc}, optional
        Default coordinates to represent points in hyperbolic space.
    scale : int, optional
        Scale of the hyperbolic space, defined as the set of points
        in Minkowski space whose squared norm is equal to -scale.
    """

    default_coords_type = 'extrinsic'
    default_point_type = 'vector'

    def __init__(self, dimension, embedding_manifold=None, scale=1,):
        assert isinstance(dimension, int) and dimension > 0
        super(Hyperbolic, self).__init__(
            dimension=dimension,
            embedding_manifold=embedding_manifold)
        self.point_type = Hyperbolic.default_point_type
        self.coords_type = Hyperbolic.default_point_type
        self.scale = scale
        self.metric =\
            HyperbolicMetric(self.dimension, self.scale)

        self.coords_transform = {
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

    @staticmethod
    def _extrinsic_to_extrinsic_coordinates(point):
        """Convert the parameterization of a point.

        Convert the parameterization of a point in hyperbolic space
        from its extrinsic coordinates, to its intrinsic coordinates
        in Minkowski space.

        Parameters
        ----------
        point_extrinsic : array-like, shape=[n_samples, dimension + 1]
            Point in hyperbolic space in extrinsic coordinates.

        Returns
        -------
        point_intrinsic : array-like, shape=[n_samples, dimension]
            Point in hyperbolic space in intrinsic coordinates.
        """
        return gs.to_ndarray(point, to_ndim=2)

    @staticmethod
    def _intrinsic_to_extrinsic_coordinates(point_intrinsic):
        """Convert intrinsic to extrinsic coordinates.

        Convert the parameterization of a point in hyperbolic space
        from its intrinsic coordinates, to its extrinsic coordinates
        in Minkowski space.

        Parameters
        ----------
        point_intrinsic : array-like, shape=[n_samples, dimension]
            Point in hyperbolic space in intrinsic coordinates.

        Returns
        -------
        point_extrinsic : array-like, shape=[n_samples, dimension + 1]
            Point in hyperbolic space in extrinsic coordinates.
        """
        point_intrinsic = gs.to_ndarray(point_intrinsic, to_ndim=2)

        coord_0 = gs.sqrt(1. + gs.linalg.norm(point_intrinsic, axis=-1) ** 2)
        coord_0 = gs.to_ndarray(coord_0, to_ndim=2, axis=1)

        point_extrinsic = gs.concatenate([coord_0, point_intrinsic], axis=-1)

        return point_extrinsic

    @staticmethod
    def _extrinsic_to_intrinsic_coordinates(point_extrinsic):
        """Convert extrinsic to intrinsic coordinates.

        Convert the parameterization of a point in hyperbolic space
        from its extrinsic coordinates in Minkowski space, to its
        intrinsic coordinates.

        Parameters
        ----------
        point_extrinsic : array-like, shape=[n_samples, dimension + 1]
            Point in hyperbolic space in extrinsic coordinates.

        Returns
        -------
        point_intrinsic : array-like, shape=[n_samples, dimension]
        """
        point_extrinsic = gs.to_ndarray(point_extrinsic, to_ndim=2)

        point_intrinsic = point_extrinsic[:, 1:]

        return point_intrinsic

    @staticmethod
    def _extrinsic_to_ball_coordinates(point):
        """Convert extrinsic to ball coordinates.

        Convert the parameterization of a point in hyperbolic space
        from its intrinsic coordinates, to the poincare ball model
        coordinates.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension + 1]
            Point in hyperbolic space in extrinsic coordinates.

        Returns
        -------
        point_ball : array-like, shape=[n_samples, dimension]
            Point in hyperbolic space in Poincare ball coordinates.
        """
        return point[:, 1:] / (1 + point[:, :1])

    @staticmethod
    def _ball_to_extrinsic_coordinates(point):
        """Convert ball to extrinsic coordinates.

        Convert the parameterization of a point in hyperbolic space
        from its poincare ball model coordinates, to the extrinsic
        coordinates.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension]
            Point in hyperbolic space in Poincare ball coordinates.

        Returns
        -------
        extrinsic : array-like, shape=[n_samples, dimension + 1]
            Point in hyperbolic space in extrinsic coordinates.
        """
        squared_norm = gs.sum(point**2, -1)
        denominator = 1 - squared_norm
        t = gs.to_ndarray((1 + squared_norm) / denominator, to_ndim=2, axis=1)
        expanded_denominator = gs.expand_dims(denominator, -1)
        expanded_denominator = gs.repeat(
            expanded_denominator, point.shape[-1], -1)
        intrinsic = (2 * point) / expanded_denominator
        return gs.concatenate([t, intrinsic], -1)

    @staticmethod
    def _half_plane_to_extrinsic_coordinates(point):
        """Convert half plane to extrinsic coordinates.

        Convert the parameterization of a point in the hyperbolic plane
        from its upper half plane model coordinates, to the extrinsic
        coordinates.

        Parameters
        ----------
        point : array-like, shape=[n_samples, 2]
            Point in hyperbolic space in half-plane coordinates.

        Returns
        -------
        extrinsic : array-like, shape=[n_samples, 3]
            Point in hyperbolic plane in extrinsic coordinates.
        """
        assert point.shape[-1] == 2
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
    def _extrinsic_to_half_plane_coordinates(point):
        """Convert extrinsic to half plane coordinates.

        Convert the parameterization of a point in the hyperbolic plane
        from its intrinsic coordinates, to the poincare upper half plane
        coordinates.

        Parameters
        ----------
        point : array-like, shape=[n_samples, 2]
            Point in the hyperbolic plane in intrinsic coordinates.

        Returns
        -------
        point_half_plane : array-like, shape=[n_samples, 2]
            Point in the hyperbolic plane in Poincare upper half-plane
            coordinates.
        """
        point_ball = \
            Hyperbolic._extrinsic_to_ball_coordinates(point)
        assert point_ball.shape[-1] == 2
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

    def to_coordinates(self, point, to_coords_type='ball'):
        """Convert coordinates of a point.

        Convert the parameterization of a point in the hyperbolic space
        from current coordinate system to the coordinate system given.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension]
                            or shape=[n_samples, dimension + 1]
            Point in hyperbolic space.
        to_point_type : str, {'extrinsic', 'intrinsic', etc}, optional
            Coordinates type.

        Returns
        -------
        point_to : array-like, shape=[n_samples, dimension]
                               or shape=[n_sample, dimension + 1]
            Point in hyperbolic space in coordinates given by to_point_type.
        """
        point = gs.to_ndarray(point, to_ndim=2, axis=0)
        if self.coords_type == to_coords_type:
            return point
        else:
            extrinsic =\
                self.coords_transform[self.coords_type + '-extrinsic'
                                      ](point)
            return self.coords_transform[
                'extrinsic-' + to_coords_type
            ](extrinsic)

    def from_coordinates(self, point, from_coords_type):
        """Convert to a type of coordinates given some type.

        Convert the parameterization of a point in hyperbolic space
        from given coordinate system to the current coordinate system.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension]
                            or shape=[n_samples, dimension + 1]
            Point in hyperbolic space in coordinates from_point_type.
        from_point_type : str, {'ball', 'extrinsic', 'intrinsic', 'half_plane'}
            Coordinates type.

        Returns
        -------
        point_current : array-like, shape=[n_samples, dimension]
                                    or shape=[n_sample, dimension + 1]
            Point in hyperbolic space.
        """
        point = gs.to_ndarray(point, to_ndim=2, axis=0)
        if self.coords_type == from_coords_type:
            return point
        else:
            extrinsic =\
                self.coords_transform[from_coords_type + '-extrinsic'](point)
            return self.coords_transform['extrinsic-' + self.coords_type
                                         ](extrinsic)

    def random_uniform(self, n_samples=1, bound=1.):
        """Sample over the hyperbolic space using uniform distribution.

        Sample over the hyperbolic space. The sampling is performed
        by sampling over uniform distribution, the sampled examples
        are considered in the intrinsic coordinates system.
        The function then transforms intrinsic samples into system
        coordinate selected.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples.
        bound: float, optional
            Bound defining the hypersquare in which to sample uniformly.

        Returns
        -------
        samples : array-like, shape=[n_samples, dimension + 1]
            Samples in hyperbolic space.
        """
        size = (n_samples, self.dimension)
        samples = bound * 2. * (gs.random.rand(*size) - 0.5)

        return self.coords_transform['intrinsic-' + self.coords_type](samples)


class HyperbolicMetric(RiemannianMetric):
    """Class that defines operations using a hyperbolic metric.

    Parameters
    ----------
    dimension : int
        Dimension of the hyperbolic space.
    point_type : str, {'extrinsic', 'intrinsic', etc}, optional
        Default coordinates to represent points in hyperbolic space.
    scale : int, optional
        Scale of the hyperbolic space, defined as the set of points
        in Minkowski space whose squared norm is equal to -scale.
    """

    default_point_type = 'vector'
    default_coords_type = 'extrinsic'

    def __init__(self, dimension, scale=1):
        super(HyperbolicMetric, self).__init__(
            dimension=dimension,
            signature=(dimension, 0, 0))
        self.point_type = HyperbolicMetric.default_point_type

        assert scale > 0, 'The scale should be strictly positive'
        self.scale = scale

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the inner-product of two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[n_samples, dimension + 1]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[n_samples, dimension + 1]
            Second tangent vector at base point.
        base_point : array-like, shape=[n_samples, dimension + 1], optional
            Point in hyperbolic space.

        Returns
        -------
        inner_prod : array-like, shape=[n_samples, 1]
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
        vector : array-like, shape=[n_samples, dimension + 1]
            Vector on the tangent space of the hyperbolic space at base point.
        base_point : array-like, shape=[n_samples, dimension + 1], optional
            Point in hyperbolic space in extrinsic coordinates.

        Returns
        -------
        sq_norm : array-like, shape=[n_samples, 1]
            Squared norm of the vector.
        """
        sq_norm = self._squared_norm(vector)
        sq_norm *= self.scale ** 2
        return sq_norm

    def _squared_norm(self, vector, base_point=None):
        """Squared norm with hyperbolic scale 1
        """
        return super().squared_norm(vector, base_point=base_point)

    def _inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Inner product with hyperbolic scale 1
        """
        return super().inner_product(tangent_vec_a, tangent_vec_b,
                                     base_point=base_point)
