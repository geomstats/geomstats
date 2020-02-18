"""The n-dimensional hyperbolic space.

The n-dimensional hyperbolic space embedded in (n+1)-dimensional
Minkowski space.
=======
"""

import logging
import math


import geomstats.backend as gs
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.minkowski import MinkowskiMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric

TOLERANCE = 1e-6

SINH_TAYLOR_COEFFS = [0., 1.,
                      0., 1 / math.factorial(3),
                      0., 1 / math.factorial(5),
                      0., 1 / math.factorial(7),
                      0., 1 / math.factorial(9)]
COSH_TAYLOR_COEFFS = [1., 0.,
                      1 / math.factorial(2), 0.,
                      1 / math.factorial(4), 0.,
                      1 / math.factorial(6), 0.,
                      1 / math.factorial(8), 0.]
INV_SINH_TAYLOR_COEFFS = [0., - 1. / 6.,
                          0., + 7. / 360.,
                          0., - 31. / 15120.,
                          0., + 127. / 604800.]
INV_TANH_TAYLOR_COEFFS = [0., + 1. / 3.,
                          0., - 1. / 45.,
                          0., + 2. / 945.,
                          0., -1. / 4725.]

EPSILON = 1e-5


class Hyperbolic(EmbeddedManifold):
    """Class for the n-dimensional Hyperbolic space.

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
        Default coordinates to represent points on the hyperbolic space.

    scale : int, optional
        Scale of the hyperbolic space, defined as the set of points
        in Minkowski space whose squared norm is equal to -scale.
    """

    def __init__(self, dimension, point_type='extrinsic', scale=1):
        assert isinstance(dimension, int) and dimension > 0
        super(Hyperbolic, self).__init__(
            dimension=dimension,
            embedding_manifold=Minkowski(dimension + 1))
        self.embedding_metric = self.embedding_manifold.metric
        self.point_type = point_type
        self.scale = scale
        self.metric = HyperbolicMetric(self.dimension, point_type, self.scale)

        self.transform_to = {
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
        self.belongs_to = {
            'ball': Hyperbolic._belongs_ball
        }

    @staticmethod
    def _belongs_ball(point, tolerance=TOLERANCE):
        """Evaluate if a point belongs to the Hyperbolic space (poin. ball).

        Test if a point belongs to the hyperbolic space based on
        the poincare ball representation, i.e. evaluate if its
        squared norm is lower than one.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension]
            Points to be tested.
        tolerance : float, optional
            Tolerance at which to evaluate how close the squared norm
            is to the reference value.

        Returns
        -------
        belongs : array-like, shape=[n_samples, 1]
            Array of booleans indicating whether the corresponding points
            belong to the hyperbolic space.
        """
        return gs.sum(point**2, -1) < (1 + tolerance)

    def belongs(self, point, tolerance=TOLERANCE):
        """Evaluate if a point belongs to the Hyperbolic space.

        Evaluate if a point belongs to the Hyperbolic space according
        to the current representation

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension] or
                shape=[n_samples, dimension + 1] for extrinsic
                coordinates
                Input points.
        tolerance : float, optional
            Tolerance at which to evaluate how close is the squared norm
            compared to the reference value.

        Returns
        -------
        belongs : array-like, shape=[n_samples, 1]
            Array of booleans evaluating if the corresponding points
            belong to the hyperbolic space.
        """
        if self.point_type == 'ball':
            return self.belongs_to[self.point_type](point, tolerance=tolerance)
        else:
            point = gs.to_ndarray(point, to_ndim=2)
            _, point_dim = point.shape
            if point_dim is not self.dimension + 1:
                if point_dim is self.dimension:
                    logging.warning(
                        'Use the extrinsic coordinates to '
                        'represent points in the Hyperbolic space.')
                    return gs.array([[False]])

            sq_norm = self.embedding_metric.squared_norm(point)
            euclidean_sq_norm = gs.linalg.norm(point, axis=-1) ** 2
            euclidean_sq_norm = gs.to_ndarray(euclidean_sq_norm,
                                              to_ndim=2, axis=1)
            diff = gs.abs(sq_norm + 1)
            belongs = diff < tolerance * euclidean_sq_norm
            return belongs

    def regularize(self, point):
        """Regularize a point to the canonical representation.

        Regularize a point to the canonical representation chosen
        for the Hyperbolic space, to avoid numerical issues.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension + 1]
                Input points.

        Returns
        -------
        projected_point : array-like, shape=[n_samples, dimension + 1]
            Point on the hyperbolic space in canonical representation
            in extrinsic coordinates.
        """
        point = gs.to_ndarray(point, to_ndim=2)

        sq_norm = self.embedding_metric.squared_norm(point)
        real_norm = gs.sqrt(gs.abs(sq_norm))

        mask_0 = gs.isclose(real_norm, 0.)
        mask_not_0 = ~mask_0
        mask_not_0_float = gs.cast(mask_not_0, gs.float32)
        projected_point = point

        projected_point = mask_not_0_float * (
            point / real_norm)
        return projected_point

    def projection_to_tangent_space(self, vector, base_point):
        """Project a vector in Minkowski space.

        Project a vector in Minkowski space on the tangent space
        of the Hyperbolic space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[n_samples, dimension + 1]
            Vector in Minkowski space to be projected.

        base_point : array-like, shape=[n_samples, dimension + 1]
            Point on the hyperbolic space.

        Returns
        -------
        tangent_vec : array-like, shape=[n_samples, dimension + 1]
            Tangent vector at the base point, equal to the projection of
            the vector in Minkowski space.
        """
        vector = gs.to_ndarray(vector, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)

        sq_norm = self.embedding_metric.squared_norm(base_point)
        inner_prod = self.embedding_metric.inner_product(base_point,
                                                         vector)

        coef = inner_prod / sq_norm
        tangent_vec = vector - gs.einsum('ni,nj->nj', coef, base_point)
        return tangent_vec

    def intrinsic_to_extrinsic_coords(self, point_intrinsic):
        """Convert the parameterization of a point.

        Convert the parameterization of a point on the hyperbolic
        space from its intrinsic coordinates to its extrinsic coordinates
        in Minkowski space.

        Parameters
        ----------
        point_intrinsic : array-like, shape=[n_samples, dimension]
            Point on the hyperbolic space in intrinsic coordinates.

        Returns
        -------
        point_extrinsic : array-like, shape=[n_samples, dimension + 1]
            Point on the hyperbolic space in extrinsic coordinates.
        """
        return Hyperbolic._intrinsic_to_extrinsic_coordinates(
            point_intrinsic)

    def extrinsic_to_intrinsic_coords(self, point_extrinsic):
        """Convert the parameterization of a point.

        Convert the parameterization of a point on the hyperbolic space
        from its extrinsic coordinates, to its intrinsic coordinates
        in Minkowski space.

        Parameters
        ----------
        point_extrinsic : array-like, shape=[n_samples, dimension + 1]
            Point on the hyperbolic space in extrinsic coordinates.

        Returns
        -------
        point_intrinsic : array-like, shape=[n_samples, dimension]
            Point on the hyperbolic space in intrinsic coordinates.
        """
        return Hyperbolic._extrinsic_to_intrinsic_coordinates(
            point_extrinsic)

    @staticmethod
    def _extrinsic_to_extrinsic_coordinates(point):
        return gs.to_ndarray(point, to_ndim=2)

    @staticmethod
    def _intrinsic_to_extrinsic_coordinates(point_intrinsic):
        """Convert point parameterization from intrinsic to extrensic coords.

        Convert the parameterization of a point on the hyperbolic space
        from its intrinsic coordinates, to its extrinsic coordinates
        in Minkowski space.

        Parameters
        ----------
        point_intrinsic : array-like, shape=[n_samples, dimension]

        Returns
        -------
        point_extrinsic : array-like, shape=[n_samples, dimension + 1]
            Point on the hyperbolic space in extrinsic coordinates
        """
        point_intrinsic = gs.to_ndarray(point_intrinsic, to_ndim=2)

        coord_0 = gs.sqrt(1. + gs.linalg.norm(point_intrinsic, axis=-1) ** 2)
        coord_0 = gs.to_ndarray(coord_0, to_ndim=2, axis=1)

        point_extrinsic = gs.concatenate([coord_0, point_intrinsic], axis=-1)

        return point_extrinsic

    @staticmethod
    def _extrinsic_to_intrinsic_coordinates(point_extrinsic):
        """Convert point parameterization from extrensic to intrinsic coords.

        Convert the parameterization of a point on the hyperbolic space
        from its extrinsic coordinates in Minkowski space, to its
        intrinsic coordinates.

        Parameters
        ----------
        point_extrinsic : array-like, shape=[n_samples, dimension + 1]

        Returns
        -------
        point_intrinsic : array-like, shape=[n_samples, dimension]
        """
        point_extrinsic = gs.to_ndarray(point_extrinsic, to_ndim=2)

        point_intrinsic = point_extrinsic[:, 1:]

        return point_intrinsic

    @staticmethod
    def _extrinsic_to_ball_coordinates(point):
        """Convert point parameterization from extrensic to ball coordinates.

        Convert the parameterization of a point on the hyperbolic space
        from its intrinsic coordinates, to the poincare ball model
        coordinates.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension + 1]
            Points on the hyperblic space in extrinsic coordinates.

        Returns
        -------
        point_ball : array-like, shape=[n_samples, dimension]
            Points on the Poincare ball coordinates.
        """
        return point[:, 1:] / (1 + point[:, :1])

    @staticmethod
    def _ball_to_extrinsic_coordinates(point):
        """Convert point parameterization from ball to extrensic coordinates.

        Convert the parameterization of a point on the hyperbolic space
        from its poincare ball model coordinates, to the extrinsic
        coordinates.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension]
            Points on the hyperbolic space in Poincare ball coordinates.

        Returns
        -------
        extrinsic : array-like, shape=[n_samples, dimension + 1]
            Point on the hyperbolic space in extrinsic coordinates.
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
        """Convert point parameterization from half-plane to extrinsic coords.

        Convert the parameterization of a point on the hyperbolic space
        from its upper half plane model coordinates, to the extrinsic
        coordinates.

        Parameters
        ----------
        point : array-like, shape=[n_samples, 2]
            Points on the hyperbolic space in half-plane coordinates.

        Returns
        -------
        extrinsic : array-like, shape=[n_samples, dimension + 1]
            Points on the hyperbolic space in extrinsic coordinates.
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
        """Convert point parameterization from extrensic to half-plane coords.

        Convert the parameterization of a point on the hyperbolic space
        from its intrinsic coordinates, to the poincare upper half plane
        coordinates.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension + 1]
           Point on the hyperbolic space in intrinsic coordinates.

        Returns
        -------
        point_half_plane : array-like, shape=[n_samples, dimension]
            Point on the hyperbolic space in Poincare upper half-plane
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

    def to_coordinates(self, point, to_point_type='ball'):
        """Convert point parameterization from current to specified coords.

        Convert the parameterization of a point on the hyperbolic space
        from current coordinates system to the coordinates system given.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension] expected or
                shape=[n_samples, dimension + 1] for extrinsic
                coordinates only.

        to_point_type : coordinates type to transform the point, can be
                        'ball', 'extrinsic', 'intrinsic', 'half_plane'

        Returns
        -------
        point_to : array-like, shape=[n_samples, dimension]
                               or shape=[n_sample, dimension + 1]
            Point on the hyperbolic space in coordinates given by
            to_point_type.
        """
        point = gs.to_ndarray(point, to_ndim=2, axis=0)
        if self.point_type == to_point_type:
            return point
        else:
            extrinsic = self.transform_to[
                self.point_type + '-extrinsic'
            ](point)
            return self.transform_to[
                'extrinsic-' + to_point_type
            ](extrinsic)

    def from_coordinates(self, point, from_point_type):
        """Convert point parameterization from provided to current coords.

        Convert the parameterization of a point on the hyperbolic space
        from given coordinates system to the current coordinates system.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension] expected or
                shape=[n_samples, dimension + 1] for extrinsic
                coordinates only.

        from_point_type : coordinates type from transform the point, can be
                         'ball', 'extrinsic', 'intrinsic', 'half_plane'

        Returns
        -------
        point_current : array-like, shape=[n_samples, dimension + 1] or
                        shape=[n_sample, dimension]
        """
        point = gs.to_ndarray(point, to_ndim=2, axis=0)
        if self.point_type == from_point_type:
            return point
        else:
            extrinsic = self.transform_to[
                from_point_type + "-extrinsic"
            ](point)
            return self.transform_to[
                "extrinsic-" + self.point_type
            ](extrinsic)

    def random_uniform(self, n_samples=1, bound=1.):
        """Sample in the hyperbolic space with the uniform distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples.

        bound: float, optional
            Bound defining the hypersquare in which to sample uniformly.

        Returns
        -------
        samples : array-like, shape=[n_samples, dimension + 1]
            Samples on the hyperbolic space.
        """
        size = (n_samples, self.dimension)
        samples = bound * 2. * (gs.random.rand(*size) - 0.5)

        return self.intrinsic_to_extrinsic_coords(samples)


class HyperbolicMetric(RiemannianMetric):
    """Class that defines operations using a Hyperbolic metric.

    Parameters
    ----------
    dimension : int
        Dimension of the hyperbolic space.

    point_type : str, {'extrinsic', 'intrinsic', etc}, optional
        Default coordinates to represent points on the hyperbolic space.

    scale : int, optional
        Scale of the hyperbolic space, defined as the set of points
        in Minkowski space whose squared norm is equal to -scale.
    """

    def __init__(self, dimension, point_type='extrinsic', scale=1):
        super(HyperbolicMetric, self).__init__(
            dimension=dimension,
            signature=(dimension, 0, 0))
        self.embedding_metric = MinkowskiMetric(dimension + 1)
        self.point_type = point_type
        assert scale > 0, 'The scale should be strictly positive'
        self.scale = scale

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the inner product of two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[n_samples, dimension + 1]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[n_samples, dimension + 1]
            Second tangent vector at base point.
        base_point : array-like, shape=[n_samples, dimension + 1], optional
            Point on the hyperbolic space.

        Returns
        -------
        inner_prod : array-like, shape=[n_samples, 1]
            Inner-product of the two tangent vectors.
        """
        inner_prod = self.embedding_metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point)
        inner_prod *= self.scale ** 2
        return inner_prod

    def squared_norm(self, vector, base_point=None):
        """Compute the squared norm of a vector at a given base point.

        Squared norm of a vector associated with the inner product
        at the tangent space at a base point. Extrinsic base point only.

        Parameters
        ----------
        vector : array-like, shape=[n_samples, dimension + 1]
            Vector on the tangent space of the hyperbolic space at base point.

        base_point : array-like, shape=[n_samples, dimension + 1], optional
            Point on the hyperbolic space in extrinsic coordinates.

        Returns
        -------
        sq_norm : array-like, shape=[n_samples, 1]
            Squared norm of the vector.
        """
        sq_norm = self.embedding_metric.squared_norm(vector)
        sq_norm *= self.scale ** 2
        return sq_norm

    def exp(self, tangent_vec, base_point):
        """Compute Riemannian exponential of tangent vector wrt to base point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, dimension + 1]
            Tangent vector at a base point.
        base_point : array-like, shape=[n_samples, dimension + 1]
            Point on the hyperbolic space.

        Returns
        -------
        exp : array-like, shape=[n_samples, dimension + 1]
            Point on the hyperbolic space equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        if self.point_type == 'extrinsic':
            tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
            base_point = gs.to_ndarray(base_point, to_ndim=2)

            sq_norm_tangent_vec = self.embedding_metric.squared_norm(
                tangent_vec)
            norm_tangent_vec = gs.sqrt(sq_norm_tangent_vec)

            mask_0 = gs.isclose(sq_norm_tangent_vec, 0.)
            mask_0 = gs.to_ndarray(mask_0, to_ndim=1)
            mask_else = ~mask_0
            mask_else = gs.to_ndarray(mask_else, to_ndim=1)
            mask_0_float = gs.cast(mask_0, gs.float32)
            mask_else_float = gs.cast(mask_else, gs.float32)

            coef_1 = gs.zeros_like(norm_tangent_vec)
            coef_2 = gs.zeros_like(norm_tangent_vec)

            coef_1 += mask_0_float * (
                1. + COSH_TAYLOR_COEFFS[2] * norm_tangent_vec ** 2
                + COSH_TAYLOR_COEFFS[4] * norm_tangent_vec ** 4
                + COSH_TAYLOR_COEFFS[6] * norm_tangent_vec ** 6
                + COSH_TAYLOR_COEFFS[8] * norm_tangent_vec ** 8)
            coef_2 += mask_0_float * (
                1. + SINH_TAYLOR_COEFFS[3] * norm_tangent_vec ** 2
                + SINH_TAYLOR_COEFFS[5] * norm_tangent_vec ** 4
                + SINH_TAYLOR_COEFFS[7] * norm_tangent_vec ** 6
                + SINH_TAYLOR_COEFFS[9] * norm_tangent_vec ** 8)
            # This avoids dividing by 0.
            norm_tangent_vec += mask_0_float * 1.0
            coef_1 += mask_else_float * (gs.cosh(norm_tangent_vec))
            coef_2 += mask_else_float * (
                (gs.sinh(norm_tangent_vec) / (norm_tangent_vec)))

            exp = (
                gs.einsum('ni,nj->nj', coef_1, base_point)
                + gs.einsum('ni,nj->nj', coef_2, tangent_vec))

            hyperbolic_space = Hyperbolic(dimension=self.dimension)
            exp = hyperbolic_space.regularize(exp)
            return exp

        elif self.point_type == 'ball':
            norm_base_point = gs.to_ndarray(
                gs.linalg.norm(base_point, -1), 2, -1)
            norm_base_point = gs.repeat(
                norm_base_point, base_point.shape[-1], -1)
            den = 1 - norm_base_point**2

            norm_tan = gs.to_ndarray(gs.linalg.norm(
                tangent_vec, axis=-1), 2, -1)
            norm_tan = gs.repeat(norm_tan, base_point.shape[-1], -1)

            lambda_base_point = 1 / den

            direction = tangent_vec / norm_tan

            factor = gs.tanh(lambda_base_point * norm_tan)

            exp = self.mobius_add(base_point, direction * factor)

            return exp
        else:
            raise NotImplementedError(
                'exp is only implemented for ball and extrinsic')

    def log(self, point, base_point):
        """Compute Riemannian logarithm of a point wrt a base point.

        If point_type = 'poincare' then base_point belongs
        to the Poincare ball and point is a vector in the euclidean
        space of the same dimension as the ball.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension + 1]
            Point on the hyperbolic space.
        base_point : array-like, shape=[n_samples, dimension + 1]
            Point on the hyperbolic space.

        Returns
        -------
        log : array-like, shape=[n_samples, dimension + 1]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        if self.point_type == 'extrinsic':
            point = gs.to_ndarray(point, to_ndim=2)
            base_point = gs.to_ndarray(base_point, to_ndim=2)

            angle = self.dist(base_point, point) / self.scale
            angle = gs.to_ndarray(angle, to_ndim=1)
            angle = gs.to_ndarray(angle, to_ndim=2)

            mask_0 = gs.isclose(angle, 0.)
            mask_else = ~mask_0

            mask_0_float = gs.cast(mask_0, gs.float32)
            mask_else_float = gs.cast(mask_else, gs.float32)

            coef_1 = gs.zeros_like(angle)
            coef_2 = gs.zeros_like(angle)

            coef_1 += mask_0_float * (
                1. + INV_SINH_TAYLOR_COEFFS[1] * angle ** 2
                + INV_SINH_TAYLOR_COEFFS[3] * angle ** 4
                + INV_SINH_TAYLOR_COEFFS[5] * angle ** 6
                + INV_SINH_TAYLOR_COEFFS[7] * angle ** 8)
            coef_2 += mask_0_float * (
                1. + INV_TANH_TAYLOR_COEFFS[1] * angle ** 2
                + INV_TANH_TAYLOR_COEFFS[3] * angle ** 4
                + INV_TANH_TAYLOR_COEFFS[5] * angle ** 6
                + INV_TANH_TAYLOR_COEFFS[7] * angle ** 8)

            # This avoids dividing by 0.
            angle += mask_0_float * 1.

            coef_1 += mask_else_float * (angle / gs.sinh(angle))
            coef_2 += mask_else_float * (angle / gs.tanh(angle))

            log = (gs.einsum('ni,nj->nj', coef_1, point) -
                   gs.einsum('ni,nj->nj', coef_2, base_point))
            return log

        elif self.point_type == 'ball':

            add_base_point = self.mobius_add(-base_point, point)

            norm_add = gs.to_ndarray(gs.linalg.norm(
                add_base_point, axis=-1), 2, -1)
            norm_add = gs.repeat(norm_add, base_point.shape[-1], -1)
            norm_base_point = gs.to_ndarray(gs.linalg.norm(
                base_point, axis=-1), 2, -1)
            norm_base_point = gs.repeat(norm_base_point,
                                        base_point.shape[-1], -1)

            log = (1 - norm_base_point**2) * gs.arctanh(norm_add)\
                * (add_base_point / norm_add)

            mask_0 = gs.all(gs.isclose(norm_add, 0.))
            log[mask_0] = 0

            return log
        else:
            raise NotImplementedError(
                'log is only implemented for ball and extrinsic')

    def mobius_add(self, point_a, point_b):
        """Compute the mobius addition of two points.

        Mobius addition is necessary for computation of the log and exp
        using the 'poincare' representation set as point_type.

        Parameters
        ----------
        point_a : array-like, shape=[n_samples, dimension + 1]
            Point on the hyperbolic space.
        point_b : array-like, shape=[n_samples, dimension + 1]
            Point on the hyperbolic space.

        Returns
        -------
        mobius_add : array-like, shape=[n_samples, 1]
            Result of the Mobius addition.
        """
        norm_point_a = gs.sum(point_a ** 2, axis=-1,
                              keepdims=True)

        # to redefine to use autograd
        norm_point_a = gs.repeat(norm_point_a, point_a.shape[-1], -1)

        norm_point_b = gs.sum(point_b ** 2, axis=-1,
                              keepdims=True)
        norm_point_b = gs.repeat(norm_point_b, point_a.shape[-1], -1)

        sum_prod_a_b = gs.sum(point_a * point_b,
                              axis=-1, keepdims=True)

        sum_prod_a_b = gs.repeat(sum_prod_a_b, point_a.shape[-1], -1)

        add_nominator = ((1 + 2 * sum_prod_a_b + norm_point_b) * point_a +
                         (1 - norm_point_a) * point_b)

        add_denominator = (1 + 2 * sum_prod_a_b + norm_point_a * norm_point_b)

        mobius_add = add_nominator / add_denominator

        return mobius_add

    def dist(self, point_a, point_b):
        """Compute the geodesic distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[n_samples, dimension + 1]
            First point on the hyperbolic space.

        point_b : array-like, shape=[n_samples, dimension + 1]
            Second point on the hyperbolic space.

        Returns
        -------
        dist : array-like, shape=[n_samples, 1]
            Geodesic distance between the two points.
        """
        if self.point_type == 'extrinsic':

            sq_norm_a = self.embedding_metric.squared_norm(point_a)
            sq_norm_b = self.embedding_metric.squared_norm(point_b)
            inner_prod = self.embedding_metric.inner_product(point_a, point_b)

            cosh_angle = - inner_prod / gs.sqrt(sq_norm_a * sq_norm_b)
            cosh_angle = gs.clip(cosh_angle, 1.0, 1e24)

            dist = gs.arccosh(cosh_angle)
            dist *= self.scale
            return dist

        elif self.point_type == 'ball':

            point_a_norm = gs.clip(gs.sum(point_a ** 2, -1), 0., 1 - EPSILON)
            point_b_norm = gs.clip(gs.sum(point_b ** 2, -1), 0., 1 - EPSILON)

            diff_norm = gs.sum((point_a - point_b) ** 2, -1)
            norm_function = 1 + 2 * \
                diff_norm / ((1 - point_a_norm) * (1 - point_b_norm))

            dist = gs.log(norm_function + gs.sqrt(norm_function ** 2 - 1))
            dist = gs.to_ndarray(dist, to_ndim=1)
            dist = gs.to_ndarray(dist, to_ndim=2, axis=1)
            dist *= self.scale
            return dist

        else:
            raise NotImplementedError(
                'dist is only implemented for ball and extrinsic')
