"""
The n-dimensional Hyperbolic space
as embedded in (n+1)-dimensional Minkowski space.
"""

import logging
import math


import geomstats.backend as gs
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.minkowski_space import MinkowskiMetric
from geomstats.geometry.minkowski_space import MinkowskiSpace
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


class HyperbolicSpace(EmbeddedManifold):
    """
    Class for the n-dimensional Hyperbolic space
    as embedded in (n+1)-dimensional Minkowski space.

    The point_type variable allows to choose the
    representation of the points as input.

    By default, point_type is set to 'extrinsic' indicating that
    points are parameterized by their extrinsic (n+1)-coordinates.

    If point_type is set to 'ball' then points are parametrized
    by their coordinates inside the Poincare Ball (n)-coordinates.
    """

    def __init__(self, dimension, point_type='extrinsic'):
        assert isinstance(dimension, int) and dimension > 0
        super(HyperbolicSpace, self).__init__(
            dimension=dimension,
            embedding_manifold=MinkowskiSpace(dimension + 1))
        self.embedding_metric = self.embedding_manifold.metric
        self.point_type = point_type
        self.metric = HyperbolicMetric(self.dimension, point_type)

        self.transform_to = {
            "ball-extrinsic":
                HyperbolicSpace._ball_to_extrinsic_coordinates,
            "extrinsic-ball":
                HyperbolicSpace._extrinsic_to_ball_coordinates,
            "intrinsic-extrinsic":
                HyperbolicSpace._intrinsic_to_extrinsic_coordinates,
            "extrinsic-intrinsic":
                HyperbolicSpace._extrinsic_to_intrinsic_coordinates,
            "extrinsic-half_plane":
                HyperbolicSpace._extrinsic_to_half_plane_coordinates,
            "half_plane-extrinsic":
                HyperbolicSpace._half_plane_to_extrinsic_coordinates,
            "extrinsic-extrinsic":
                HyperbolicSpace._extrinsic_to_extrinsic_coordinates
        }
        self.belongs_to = {
            "ball": HyperbolicSpace._belongs_ball,
            "extrinsic": HyperbolicSpace._belong_extrinsic,
            "intrinsic": HyperbolicSpace._belong_extrinsic,
            "half_plane": HyperbolicSpace._belong_extrinsic
        }

    @staticmethod
    def _belongs_ball(point, tolerance=TOLERANCE):
        """
        Evaluate if a point belongs to the Hyperbolic space, based on
        poincare ball representationj
        i.e. evaluate if its squared norm is lower than one

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension]
                Input points.
        tolerance : float, optional

        Returns
        -------
        belongs : array-like, shape=[n_samples, 1]
        """

        return gs.sum(point**2, -1) < (1 + tolerance)

    def belongs(self, point, tolerance=TOLERANCE):
        """
        Evaluate if a point belongs to the Hyperbolic space,
        according to the current representation

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension] or
                shape=[n_samples, dimension + 1] for extrinsic
                coordinates
                Input points.
        tolerance : float, optional

        Returns
        -------
        belongs : array-like, shape=[n_samples, 1]
        """
        if(self.point_type == 'ball'):
            return self.belongs_to[self.point_type](point, tolerance=tolerance)
        else:
            point = gs.to_ndarray(point, to_ndim=2)
            _, point_dim = point.shape
            if point_dim is not self.dimension + 1:
                if point_dim is self.dimension:
                    logging.warning(
                        'Use the extrinsic coordinates to '
                        'represent points on the hyperbolic space.')
                    return gs.array([[False]])

            sq_norm = self.embedding_metric.squared_norm(point)
            euclidean_sq_norm = gs.linalg.norm(point, axis=-1) ** 2
            euclidean_sq_norm = gs.to_ndarray(euclidean_sq_norm,
                                              to_ndim=2, axis=1)
            diff = gs.abs(sq_norm + 1)
            belongs = diff < tolerance * euclidean_sq_norm
            return belongs

    def regularize(self, point):
        """
        Regularize a point to the canonical representation
        chosen for the Hyperbolic space, to avoid numerical issues.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension + 1]
                Input points.

        Returns
        -------
        projected_point : array-like, shape=[n_samples, dimension + 1]
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
        """
        Project a vector in Minkowski space
        on the tangent space of the Hyperbolic space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[n_samples, dimension + 1]
        base_point : array-like, shape=[n_samples, dimension + 1]

        Returns
        -------
        tangent_vec : array-like, shape=[n_samples, dimension + 1]
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
        """
        Convert the parameterization of a point on the Hyperbolic space
        from its intrinsic coordinates, to its extrinsic coordinates
        in Minkowski space.

        Parameters
        ----------
        point_intrinsic : array-like, shape=[n_samples, dimension]

        Returns
        -------
        point_extrinsic : array-like, shape=[n_samples, dimension + 1]
        """
        return HyperbolicSpace._intrinsic_to_extrinsic_coordinates(
            point_intrinsic)

    def extrinsic_to_intrinsic_coords(self, point_extrinsic):
        """
        Convert the parameterization of a point on the Hyperbolic space
        from its extrinsic coordinates, to its intrinsic coordinates
        in Minkowski space.

        Parameters
        ----------
        point_intrinsic : array-like, shape=[n_samples, dimension + 1]

        Returns
        -------
        point_extrinsic : array-like, shape=[n_samples, dimension]
        """
        return HyperbolicSpace._extrinsic_to_intrinsic_coordinates(
            point_extrinsic)

    @staticmethod
    def _extrinsic_to_extrinsic_coordinates(point):
        return gs.to_ndarray(point, to_ndim=2)

    @staticmethod
    def _intrinsic_to_extrinsic_coordinates(point_intrinsic):
        """
        Convert the parameterization of a point on the Hyperbolic space
        from its intrinsic coordinates, to its extrinsic coordinates
        in Minkowski space.

        Parameters
        ----------
        point_intrinsic : array-like, shape=[n_samples, dimension]

        Returns
        -------
        point_extrinsic : array-like, shape=[n_samples, dimension + 1]
        """
        point_intrinsic = gs.to_ndarray(point_intrinsic, to_ndim=2)

        coord_0 = gs.sqrt(1. + gs.linalg.norm(point_intrinsic, axis=-1) ** 2)
        coord_0 = gs.to_ndarray(coord_0, to_ndim=2, axis=1)

        point_extrinsic = gs.concatenate([coord_0, point_intrinsic], axis=-1)

        return point_extrinsic

    @staticmethod
    def _extrinsic_to_intrinsic_coordinates(point_extrinsic):
        """
        Convert the parameterization of a point on the Hyperbolic space
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
        """
        Convert the parameterization of a point on the Hyperbolic space
        from its intrinsic coordinates, to the poincare ball model
        coordinates.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension + 1] in intrinsic
                coordinates

        Returns
        -------
        point_ball : array-like, shape=[n_samples, dimension] in
                     poincare ball coordinates
        """
        return point[:, 1:] / (1 + point[:, :1])

    @staticmethod
    def _ball_to_extrinsic_coordinates(point):
        """
        Convert the parameterization of a point on the Hyperbolic space
        from its poincare ball model coordinates, to the extrinsic
        coordinates.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension] in Poincare ball
                coordinates

        Returns
        -------
        point_ball : array-like, shape=[n_samples, dimension + 1] in
                     extrinsic coordinate
        """
        squared_norm = gs.sum(point**2, -1)
        denominator = 1-squared_norm
        t = gs.to_ndarray((1+squared_norm)/denominator, to_ndim=2, axis=1)
        expanded_denominator = gs.expand_dims(denominator, -1)
        expanded_denominator = gs.repeat(expanded_denominator,
                                         point.shape[-1], -1)
        intrasic = (2*point)/expanded_denominator
        return gs.concatenate([t, intrasic], -1)

    @staticmethod
    def _half_plane_to_extrinsic_coordinates(point):
        """
        Convert the parameterization of a point on the Hyperbolic space
        from its upper half plane model coordinates, to the extrinsic
        coordinates.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension] in Poincare ball
                coordinates

        Returns
        -------
        point_ball : array-like, shape=[n_samples, dimension + 1] in
                     extrinsic coordinate
        """
        assert(point.shape[-1] == 2)
        x, y = point[:, 0], point[:, 1]
        x2 = point[:, 0]**2
        den = x2 + (1+y)**2
        x = gs.to_ndarray(x, to_ndim=2, axis=0)
        y = gs.to_ndarray(y, to_ndim=2, axis=0)
        x2 = gs.to_ndarray(x2, to_ndim=2, axis=0)
        den = gs.to_ndarray(den, to_ndim=2, axis=0)
        ball_point = gs.hstack((2*x/den, (x2+y**2-1)/den))
        return HyperbolicSpace._ball_to_extrinsic_coordinates(ball_point)

    @staticmethod
    def _extrinsic_to_half_plane_coordinates(point):
        """
        Convert the parameterization of a point on the Hyperbolic space
        from its intrinsic coordinates, to the poincare upper half plane
        coordinates.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension + 1] in intrinsic
                coordinates

        Returns
        -------
        point_ball : array-like, shape=[n_samples, dimension] in
                     poincare ball coordinates
        """
        point_ball = \
            HyperbolicSpace._extrinsic_to_ball_coordinates(point)
        assert(point_ball.shape[-1] == 2)
        x, y = point_ball[:, 0], point_ball[:, 1]
        x2 = point_ball[:, 0]**2
        den = x2 + (1-y)**2

        x = gs.to_ndarray(x, to_ndim=2, axis=0)
        y = gs.to_ndarray(y, to_ndim=2, axis=0)
        x2 = gs.to_ndarray(x2, to_ndim=2, axis=0)
        den = gs.to_ndarray(den, to_ndim=2, axis=0)

        return gs.hstack(((2*x)/den, (1-x2-y**2)/den))

    def to_coordinates(self, point, to_point_type='ball'):
        """
        Convert the parameterization of a point on the Hyperbolic space
        from current coordinates system to the coordinates system given

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension] expected or
                shape=[n_samples, dimension + 1] for extrinsic
                coordinates only

        to_point_type : coordinates type to transform the point, can be
                        'ball', 'extrinsic', 'intrinsic'

        Returns
        -------
        point_ball : array-like, shape=[n_samples, dimension + 1] in
                     extrinsic coordinate
        """
        point = gs.to_ndarray(point, to_ndim=2, axis=0)
        if self.point_type == to_point_type:
            return point
        else:
            extrinsic = self.transform_to[
                self.point_type+"-extrinsic"
                ](point)
            return self.transform_to[
                "extrinsic-"+to_point_type
                ](extrinsic)

    def from_coordinates(self, point, from_point_type):
        """
        Convert the parameterization of a point on the Hyperbolic space
        from given coordinates system to the current coordinates system

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension] expected or
                shape=[n_samples, dimension + 1] for extrinsic
                coordinates only

        to_point_type : coordinates type to transform the point, can be
                        'ball', 'extrinsic', 'intrinsic'

        Returns
        -------
        point_ball : array-like, shape=[n_samples, dimension + 1] in
                     extrinsic coordinate
        """
        point = gs.to_ndarray(point, to_ndim=2, axis=0)
        if self.point_type == from_point_type:
            return point
        else:
            extrinsic = self.transform_to[
                from_point_type+"-extrinsic"
                ](point)
            return self.transform_to[
                "extrinsic-"+self.point_type
                ](extrinsic)

    def random_uniform(self, n_samples=1, bound=1.):
        """
        Sample in the Hyperbolic space with the uniform distribution.

        Parameters
        ----------
        n_samples : int, optional
        bound: float, optional

        Returns
        -------
        point : array-like, shape=[n_samples, dimension + 1]
        """

        size = (n_samples, self.dimension)
        point = bound * 2. * (gs.random.rand(*size) - 0.5)

        return self.intrinsic_to_extrinsic_coords(point)


class HyperbolicMetric(RiemannianMetric):

    def __init__(self, dimension, point_type='extrinsic'):
        super(HyperbolicMetric, self).__init__(
            dimension=dimension,
            signature=(dimension, 0, 0))
        self.embedding_metric = MinkowskiMetric(dimension + 1)
        self.point_type = point_type

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """
        Inner product.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[n_samples, dimension + 1]
                                    or shape=[1, dimension + 1]
        tangent_vec_b : array-like, shape=[n_samples, dimension + 1]
                                    or shape=[1, dimension + 1]
        base_point : array-like, shape=[n_samples, dimension + 1]
                                 or shape=[1, dimension + 1]

        Returns
        -------
        inner_prod : array-like, shape=[n_samples, 1]
                                 or shape=[1, 1]
        """
        inner_prod = self.embedding_metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point)
        return inner_prod

    def squared_norm(self, vector, base_point=None):
        """
        Squared norm of a vector associated with the inner product
        at the tangent space at a base point. Extrinsic base point only

        Parameters
        ----------
        vector : array-like, shape=[n_samples, dimension + 1]
                             or shape=[1, dimension + 1]
        base_point : array-like, shape=[n_samples, dimension + 1]
                                 or shape=[1, dimension + 1]

        Returns
        -------
        sq_norm : array-like, shape=[n_samples, 1]
                              or shape=[1, 1]
        """
        sq_norm = self.embedding_metric.squared_norm(vector)
        return sq_norm

    def exp(self, tangent_vec, base_point):
        """
        Riemannian exponential of a tangent vector wrt to a base point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, dimension + 1]
                                  or shape=[1, dimension + 1]
        base_point : array-like, shape=[n_samples, dimension + 1]
                                 or shape=[1, dimension + 1]

        Returns
        -------
        exp : array-like, shape=[n_samples, dimension + 1]
                          or shape=[1, dimension + 1]
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

            exp = (gs.einsum('ni,nj->nj', coef_1, base_point)
                   + gs.einsum('ni,nj->nj', coef_2, tangent_vec))

            hyperbolic_space = HyperbolicSpace(dimension=self.dimension)
            exp = hyperbolic_space.regularize(exp)
            return exp

        elif self.point_type == 'ball':
            norm_base_point = base_point.norm(2,
                                              -1, keepdim=True).expand_as(
                                                base_point)

            lambda_base_point = 1 / (1 - norm_base_point ** 2)

            norm_tangent_vector = tangent_vec.norm(2,
                                                   -1, keepdim=True).expand_as(
                                                    tangent_vec)

            direction = tangent_vec / norm_tangent_vector

            factor = gs.tanh(lambda_base_point * norm_tangent_vector)

            exp = self.mobius_add(base_point, direction * factor)

            exp[norm_tangent_vector == 0] = \
                base_point[norm_tangent_vector == 0]

            return exp
        else:
            raise NotImplementedError(
                    "exp is only implemented for ball and extrinsic")

    def log(self, point, base_point):
        """
        Riemannian logarithm of a point wrt a base point.
        If point_type = 'poincare' then base_point belongs
        to the Poincare ball and point is a vector in the euclidean
        space of the same dimension as the ball.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension + 1]
                            or shape=[1, dimension + 1]
        base_point : array-like, shape=[n_samples, dimension + 1]
                                 or shape=[1, dimension + 1]

        Returns
        -------
        log : array-like, shape=[n_samples, dimension + 1]
                          or shape=[1, dimension + 1]
        """
        if self.point_type == 'extrinsic':
            point = gs.to_ndarray(point, to_ndim=2)
            base_point = gs.to_ndarray(base_point, to_ndim=2)

            angle = self.dist(base_point, point)
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

            norm_add = add_base_point.norm(2,
                                           -1, keepdim=True).expand_as(
                add_base_point)

            norm_base_point = base_point.norm(2,
                                              -1, keepdim=True).expand_as(
                add_base_point)

            res = (1 - norm_base_point ** 2) * \
                  ((gs.arc_tanh(norm_add))) * (add_base_point / norm_add)

            mask_0 = gs.all(gs.isclose(norm_add, 0))
            res[mask_0] = 0

            return res
        else:
            raise NotImplementedError(
                    "log is only implemented for ball and extrinsic")

    def mobius_add(self, point_a, point_b):
        """
                Mobius addition operation that is necessary operation
                to compute the log and exp using the 'poincare'
                representation set as point_type.

        Parameters
        ----------
        point_a : array-like, shape=[n_samples, dimension + 1]
                              or shape=[1, dimension + 1]
        point_b : array-like, shape=[n_samples, dimension + 1]
                              or shape=[1, dimension + 1]

        Returns
        -------
        mobius_add : array-like, shape=[n_samples, 1]
                           or shape=[1, 1]
        """
        norm_point_a = gs.sum(point_a ** 2, dim=-1,
                              keepdim=True).expand_as(point_a)
        norm_point_b = gs.sum(point_b ** 2, dim=-1,
                              keepdim=True).expand_as(point_a)
        sum_prod_a_b = (point_a * point_b).sum(-1,
                                               keepdim=True).expand_as(point_a)

        add_nominator = ((1 + 2 * sum_prod_a_b + norm_point_b) * point_a +
                         (1 - norm_point_a) * point_b)

        add_denominator = (1 + 2 * sum_prod_a_b + norm_point_a * norm_point_b)

        mobius_add = add_nominator/add_denominator

        return mobius_add

    def dist(self, point_a, point_b):
        """
        Geodesic distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[n_samples, dimension + 1]
                              or shape=[1, dimension + 1]
        point_b : array-like, shape=[n_samples, dimension + 1]
                              or shape=[1, dimension + 1]

        Returns
        -------
        dist : array-like, shape=[n_samples, 1]
                           or shape=[1, 1]
        """

        if self.point_type == 'extrinsic':

            sq_norm_a = self.embedding_metric.squared_norm(point_a)
            sq_norm_b = self.embedding_metric.squared_norm(point_b)
            inner_prod = self.embedding_metric.inner_product(point_a, point_b)

            cosh_angle = - inner_prod / gs.sqrt(sq_norm_a * sq_norm_b)
            cosh_angle = gs.clip(cosh_angle, 1.0, 1e24)

            dist = gs.arccosh(cosh_angle)

            return dist

        elif self.point_type == 'ball':

            point_a_norm = gs.clip(gs.sum(point_a ** 2, -1), 0., 1 - EPSILON)
            point_b_norm = gs.clip(gs.sum(point_b ** 2, -1), 0., 1 - EPSILON)
            diff_norm = gs.sum((point_a - point_b) ** 2, -1)
            norm_function = 1 + 2 * \
                diff_norm / ((1 - point_a_norm) * (1 - point_b_norm))
            dist = gs.log(norm_function + gs.sqrt(norm_function ** 2 - 1))

            return dist

        else:
            raise NotImplementedError(
                    "distance is only implemented for ball and extrinsic")
