"""Base class for the n-dimensional hyperbolic space.

The n-dimensional hyperbolic regardless its different representations.
"""

import geomstats.backend as gs
from geomstats.geometry.diffeo import Diffeo


class HyperbolicDiffeo(Diffeo):
    """Diffeomorphism between hyperbolic coordinates.

    Parameters
    ----------
    from_coordinates_system : str, {'extrinsic', 'ball', 'half-space'}
        Coordinates type.
    to_coordinates_system : str, {'extrinsic', 'ball', 'half-space'}
        Coordinates type.
    """

    def __init__(self, from_coordinates_system, to_coordinates_system):
        self.from_coordinates_system = from_coordinates_system
        self.to_coordinates_system = to_coordinates_system

    def __call__(self, base_point):
        """Diffeomorphism at base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space_shape]
            Base point.

        Returns
        -------
        image_point : array-like, shape=[..., *image_shape]
            Image point.
        """
        return _Hyperbolic.change_coordinates_system(
            base_point, self.from_coordinates_system, self.to_coordinates_system
        )

    def inverse(self, image_point):
        r"""Inverse diffeomorphism at image point.

        :math:`f^{-1}: N \rightarrow M`

        Parameters
        ----------
        image_point : array-like, shape=[..., *image_shape]
            Image point.

        Returns
        -------
        base_point : array-like, shape=[..., *space_shape]
            Base point.
        """
        return _Hyperbolic.change_coordinates_system(
            image_point, self.to_coordinates_system, self.from_coordinates_system
        )

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        r"""Tangent diffeomorphism at base point.

        df_p is a linear map from T_pM to T_f(p)N.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space_shape]
            Tangent vector at base point.
        base_point : array-like, shape=[..., *space_shape]
            Base point.
        image_point : array-like, shape=[..., *image_shape]
            Image point.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., *image_shape]
            Image tangent vector at image of the base point.
        """
        if base_point is None:
            base_point = self.inverse(image_point)
        return _Hyperbolic.change_tangent_coordinates_system(
            tangent_vec,
            base_point,
            self.from_coordinates_system,
            self.to_coordinates_system,
        )

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
        r"""Inverse tangent diffeomorphism at image point.

        df^-1_p is a linear map from T_f(p)N to T_pM

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., *image_shape]
            Image tangent vector at image point.
        image_point : array-like, shape=[..., *image_shape]
            Image point.
        base_point : array-like, shape=[..., *space_shape]
            Base point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *space_shape]
            Tangent vector at base point.
        """
        if image_point is None:
            image_point = self(base_point)

        return _Hyperbolic.change_tangent_coordinates_system(
            image_tangent_vec,
            image_point,
            self.to_coordinates_system,
            self.from_coordinates_system,
        )


def _scalarvecmul(scalar, vec):
    """Vectorized scalar vector multiplication.

    Parameters
    ----------
    scalar : array-like, shape=[...]
    vec : array-like, shape=[..., dim]
    """
    return gs.einsum("...,...i->...i", scalar, vec)


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
        intrinsic = _scalarvecmul(1.0 / denominator, 2 * point)
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
        component_1 = _scalarvecmul(2.0 / den, point[..., :-1])
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
        component_1 = _scalarvecmul(2.0 / den, point[..., :-1])
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
        component_1 = _scalarvecmul(
            2.0 / den, tangent_vec[..., :-1]
        ) - 4.0 * _scalarvecmul(
            (scalar_prod + tangent_vec[..., -1]) / den**2,
            base_point[..., :-1],
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
        tangent_vec_half_space : array-like, shape=[..., dim]
            Tangent vector in the Poincare half-space.
        """
        sq_norm = gs.sum(base_point**2, axis=-1)
        den = 1 + sq_norm - 2 * base_point[..., -1]
        scalar_prod = gs.sum(base_point * tangent_vec, -1)
        component_1 = _scalarvecmul(
            2.0 / den, tangent_vec[..., :-1]
        ) - 4.0 * _scalarvecmul(
            (scalar_prod - tangent_vec[..., -1]) / den**2,
            base_point[..., :-1],
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
    def extrinsic_to_ball_tangent(tangent_vec, base_point):
        """Convert extrinsic to ball tangent coordinates.

        Convert the parameterization of a tangent vector to the
        hyperbolic space from its extrinsic coordinates, to the
        Poincare ball coordinates.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim + 1]
            Tangent vector at the base point in extrinsic coordinates.
        base_point : array-like, shape=[..., dim + 1]
            Point in extrinsic coordinates.

        Returns
        -------
        tangent_vec_ball : array-like, shape=[..., dim]
            Tangent vector in the Poincare ball.
        """
        den = 1 + base_point[..., 0]
        return -_scalarvecmul(
            tangent_vec[..., 0] / den**2,
            base_point[..., 1:],
        ) + _scalarvecmul(1 / den, tangent_vec[..., 1:])

    @staticmethod
    def ball_to_extrinsic_tangent(tangent_vec, base_point):
        """Convert ball to extrinsic tangent coordinates.

        Convert the parameterization of a tangent vector to the
        hyperbolic space from its Poincare ball coordinates, to the
        extrinsic coordinates.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at the base point in the Poincare ball.
        base_point : array-like, shape=[..., dim]
            Point in the Poincare ball.

        Returns
        -------
        tangent_vec_extrinsic : array-like, shape=[..., dim + 1]
            Tangent vector in extrinsic coordinates.
        """
        sq_norm = gs.sum(base_point**2, axis=-1)
        scalar_prod = gs.sum(tangent_vec * base_point, axis=-1)
        den = (1 - sq_norm) ** 2

        term_1 = 2 * _scalarvecmul(1 - sq_norm, tangent_vec)
        term_2 = 4 * _scalarvecmul(scalar_prod, base_point)
        return _scalarvecmul(
            1 / den,
            gs.concatenate([4 * scalar_prod[..., None], term_1 + term_2], axis=-1),
        )

    @classmethod
    def half_space_to_extrinsic_tangent(cls, tangent_vec, base_point):
        """Convert half-space to extrinsinc tangent coordinates.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at the base point in the Poincare half-space.
        base_point : array-like, shape=[..., dim]
            Point in the Poincare half-space.

        Returns
        -------
        tangent_vec_extrinsic : array-like, shape=[..., dim + 1]
            Tangent vector in extrinsic coordinates.
        """
        tangent_vec_ball = cls.half_space_to_ball_tangent(tangent_vec, base_point)
        base_point_ball = cls.half_space_to_ball_coordinates(base_point)
        return cls.ball_to_extrinsic_tangent(tangent_vec_ball, base_point_ball)

    @classmethod
    def extrinsic_to_half_space_tangent(cls, tangent_vec, base_point):
        """Convert half-space to extrinsinc tangent coordinates.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim + 1]
            Tangent vector in extrinsic coordinates.
        base_point : array-like, shape=[..., dim + 1]
            Point in extrinsic coordinates.

        Returns
        -------
        tangent_vec_half_space : array-like, shape=[..., dim ]
            Tangent vector at the base point in the Poincare half-space.
        """
        tangent_vec_ball = cls.extrinsic_to_ball_tangent(tangent_vec, base_point)
        base_point_ball = cls._extrinsic_to_ball_coordinates(base_point)
        return cls.ball_to_half_space_tangent(tangent_vec_ball, base_point_ball)

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

    @staticmethod
    def change_tangent_coordinates_system(
        tangent_vec, base_point, from_coordinates_system, to_coordinates_system
    ):
        """Convert coordinates of a tangent vector.

        Convert the parameterization of a tangent vector in the hyperbolic space
        from current given coordinate system to an other also given in parameters.
         The possible coordinates system are 'extrinsic', 'ball' and
        'half-plane' that correspond respectively to extrinsic coordinates in the
        hyperboloid, coordinates in the Poincare ball model and coordinates in the
        Poincare upper half-space model.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {dim, dim + 1}]
            Tangent vector in hyperbolic space.
        base_point : array-like, shape=[..., {dim, dim + 1}]
            Point in hyperbolic space.
        from_coordinates_system : str, {'extrinsic', 'intrinsic', etc}
            Coordinates type.
        to_coordinates_system : str, {'extrinsic', 'intrinsic', etc}
            Coordinates type.

        Returns
        -------
        tangent_vec_to : array-like, shape=[..., dim]
                               or shape=[n_sample, dim + 1]
            Tangent vector in hyperbolic space in coordinates given by
            to_coordinates_system.
        """
        coords_transform = {
            "ball-extrinsic": _Hyperbolic.ball_to_extrinsic_tangent,
            "extrinsic-ball": _Hyperbolic.extrinsic_to_ball_tangent,
            "ball-half-space": _Hyperbolic.ball_to_half_space_tangent,
            "half-space-ball": _Hyperbolic.half_space_to_ball_tangent,
            "extrinsic-half-space": _Hyperbolic.extrinsic_to_half_space_tangent,
            "half-space-extrinsic": _Hyperbolic.half_space_to_extrinsic_tangent,
        }

        if from_coordinates_system == to_coordinates_system:
            return gs.copy(tangent_vec)

        func = coords_transform.get(
            f"{from_coordinates_system}-{to_coordinates_system}"
        )
        return func(tangent_vec, base_point)

    def to_coordinates(self, point, to_coords_type="ball"):
        """Convert point to a target coordinate system.

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
        return self.change_coordinates_system(point, self.coords_type, to_coords_type)

    def from_coordinates(self, point, from_coords_type):
        """Convert point to the current coordinate system.

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
        return self.change_coordinates_system(point, from_coords_type, self.coords_type)

    def to_tangent_coordinates(self, tangent_vec, base_point, to_coords_type):
        """Convert tangent vector to a target coordinate system.

        Convert the parameterization of a tangent vector in the hyperbolic space
        from current coordinate system to the coordinate system given.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {dim, dim + 1}]
            Tangent vector to hyperbolic space.
        base_point : array-like, shape=[..., {dim, dim + 1}]
            Point in hyperbolic space.
        to_coords_type : str, {'extrinsic', 'half-space', 'ball'}
            Coordinates type.

        Returns
        -------
        tangent_vec_to : array-like, shape=[..., {dim, dim + 1}]
            Tangent vector in hyperbolic space in coordinates given by to_coords_type.
        """
        return self.change_tangent_coordinates_system(
            tangent_vec, base_point, self.coords_type, to_coords_type
        )

    def from_tangent_coordinates(self, tangent_vec, base_point, from_coords_type):
        """Convert tangent vector to the current coordinate system.

        Convert the parameterization of a tangent vector to hyperbolic space
        from given coordinate system to the current coordinate system.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {dim, dim + 1}]
            Tangent vector to hyperbolic space.
        base_point : array-like, shape=[..., {dim, dim + 1}]
            Point in hyperbolic space.
        from_coords_type : str, {'extrinsic', 'half-space', 'ball'}
            Coordinates type.

        Returns
        -------
        point_current : array-like, shape=[..., {dim, dim + 1}]
            Tangent vector in hyperbolic space.
        """
        return self.change_tangent_coordinates_system(
            tangent_vec, base_point, from_coords_type, self.coords_type
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

        samples = self.change_coordinates_system(samples, "intrinsic", self.coords_type)

        if n_samples == 1:
            return samples[0]
        return samples
