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
    """

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
        coord_0 = gs.sqrt(1.0 + gs.sum(point**2, axis=-1))
        return gs.concatenate([coord_0[..., None], point], axis=-1)

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
        return point[..., 1:]

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
        return point[..., 1:] / (1 + point[..., :1])

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
        intrinsic = gs.einsum("...i, ...->...i", 2 * point, 1.0 / denominator)
        return gs.concatenate([t[..., None], intrinsic], -1)

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
        return cls._ball_to_extrinsic_coordinates(point_ball)

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
        point_ball = cls._extrinsic_to_ball_coordinates(point)
        return cls.ball_to_half_space_coordinates(point_ball)

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
        sq_norm = gs.sum(point**2, axis=-1)
        den = 1 + sq_norm + 2 * point[..., -1]
        component_1 = gs.einsum("...i,...->...i", point[..., :-1], 2.0 / den)
        component_2 = (sq_norm - 1) / den
        return gs.concatenate([component_1, component_2[..., None]], axis=-1)

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
        sq_norm = gs.sum(point**2, axis=-1)
        den = 1 + sq_norm - 2 * point[..., -1]
        component_1 = gs.einsum("...i,...->...i", point[..., :-1], 2.0 / den)
        component_2 = (1 - sq_norm) / den
        return gs.concatenate([component_1, component_2[..., None]], axis=-1)

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
        sq_norm = gs.sum(base_point**2, axis=-1)
        den = 1.0 + sq_norm + 2.0 * base_point[..., -1]
        scalar_prod = gs.sum(base_point * tangent_vec, axis=-1)
        component_1 = gs.einsum(
            "...i,...->...i", tangent_vec[..., :-1], 2.0 / den
        ) - 4.0 * gs.einsum(
            "...i,...->...i",
            base_point[..., :-1],
            (scalar_prod + tangent_vec[..., -1]) / den**2,
        )
        component_2 = (
            2 * scalar_prod / den
            - 2 * (sq_norm - 1) * (scalar_prod + tangent_vec[..., -1]) / den**2
        )
        tangent_vec_ball = gs.concatenate(
            [component_1, component_2[..., None]], axis=-1
        )

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
        sq_norm = gs.sum(base_point**2, axis=-1)
        den = 1 + sq_norm - 2 * base_point[..., -1]
        scalar_prod = gs.sum(base_point * tangent_vec, -1)
        component_1 = gs.einsum(
            "...i,...->...i", tangent_vec[..., :-1], 2.0 / den
        ) - 4.0 * gs.einsum(
            "...i,...->...i",
            base_point[..., :-1],
            (scalar_prod - tangent_vec[..., -1]) / den**2,
        )
        component_2 = (
            -2.0 * scalar_prod / den
            - 2 * (1.0 - sq_norm) * (scalar_prod - tangent_vec[..., -1]) / den**2
        )
        tangent_vec_half_space = gs.concatenate(
            [component_1, component_2[..., None]], axis=-1
        )

        return tangent_vec_half_space

    @staticmethod
    def change_coordinates_system(
        point, from_coordinates_system, to_coordinates_system
    ):
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
            Point in hyperbolic space in coordinates given by
            to_coordinates_system.
        """
        coords_transform = {
            "ball-extrinsic": _Hyperbolic._ball_to_extrinsic_coordinates,
            "extrinsic-ball": _Hyperbolic._extrinsic_to_ball_coordinates,
            "ball-half-space": _Hyperbolic.ball_to_half_space_coordinates,
            "half-space-ball": _Hyperbolic.half_space_to_ball_coordinates,
            "intrinsic-extrinsic": _Hyperbolic._intrinsic_to_extrinsic_coordinates,
            "extrinsic-intrinsic": _Hyperbolic._extrinsic_to_intrinsic_coordinates,
            "extrinsic-half-space": _Hyperbolic._extrinsic_to_half_space_coordinates,
            "half-space-extrinsic": _Hyperbolic._half_space_to_extrinsic_coordinates,
        }

        if from_coordinates_system == to_coordinates_system:
            return gs.copy(point)

        func = coords_transform.get(
            f"{from_coordinates_system}-{to_coordinates_system}"
        )
        if func is not None:
            return func(point)

        extrinsic = coords_transform[f"{from_coordinates_system}-extrinsic"](point)
        return coords_transform[f"extrinsic-{to_coordinates_system}"](extrinsic)

    def to_coordinates(self, point, to_coords_type="ball"):
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
            Point in hyperbolic space in coordinates given by to_coords_type.
        """
        return self.change_coordinates_system(
            point, self.default_coords_type, to_coords_type
        )

    def from_coordinates(self, point, from_coords_type):
        """Convert to a type of coordinates given some type.

        Convert the parameterization of a point in hyperbolic space
        from given coordinate system to the current coordinate system.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, dim + 1}]
            Point in hyperbolic space in coordinates from_coords_type.
        from_coords_type : str, {'ball', 'extrinsic', 'intrinsic', ...}
            Coordinates type.

        Returns
        -------
        point_current : array-like, shape=[..., {dim, dim + 1}]
            Point in hyperbolic space.
        """
        return self.change_coordinates_system(
            point, from_coords_type, self.default_coords_type
        )

    def random_point(self, n_samples=1, bound=1.0):
        """Sample over the hyperbolic space.

        Sample over the hyperbolic space. The sampling is performed by sampling from
        the uniform distribution with respect to the intrinsic co-ordinates. This is
        not uniform with respect to the volume measure. The function then transforms
        intrinsic samples into the selected co-ordinate system.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound: float
            Bound defining the hypersquare in which to sample.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., dim + 1]
            Samples in hyperbolic space.
        """
        size = (n_samples, self.dim)
        samples = bound * 2.0 * (gs.random.rand(*size) - 0.5)

        samples = self.change_coordinates_system(
            samples, "intrinsic", self.default_coords_type
        )

        if n_samples == 1:
            return samples[0]
        return samples


class HyperbolicMetric(RiemannianMetric):
    """Class that defines operations using a hyperbolic metric.

    This class does not contain any methods and is only defined to act as a parent
    class for `HyperboloidMetric`, `PoincareBallMetric` and `PoincareHalfSpaceMetric`.
    """
