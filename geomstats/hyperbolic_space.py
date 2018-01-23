"""
Computations on the Hyperbolic space H_n
as embedded in Minkowski space R^{1,n}.

Elements of the Hyperbolic space are the elements
of Minkowski space of squared norm -1.

NB: we use "riemannian" to refer to "pseudo-riemannian".
"""

import logging
import numpy as np
import math

from geomstats.minkowski_space import MinkowskiMetric
from geomstats.manifold import Manifold
from geomstats.riemannian_metric import RiemannianMetric

EPSILON = 1e-6
TOLERANCE = 1e-12

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


class HyperbolicSpace(Manifold):
    """
    Hyperbolic space embedded in Minkowski space.
    Note: points are parameterized by the extrinsic
    coordinates by defaults.
    """

    def __init__(self, dimension):
        self.dimension = dimension
        self.metric = HyperbolicMetric(self.dimension)
        self.embedding_metric = MinkowskiMetric(self.dimension + 1)

    def belongs(self, point, tolerance=TOLERANCE):
        """
        By definition, a point on the Hyperbolic space
        has Minkowski squared norm -1.

        Note: point must be given in extrinsic coordinates.
        """
        point_dim = len(point)
        if point_dim is not self.dimension + 1:
            if point_dim is self.dimension:
                logging.warning('Use the extrinsic coordinates to '
                                'represent points on the hypersphere.')
            return False
        sq_norm = self.embedding_metric.squared_norm(point)
        return abs(sq_norm + 1.) < tolerance

    def intrinsic_to_extrinsic_coords(self, point_intrinsic):
        """
        From the intrinsic coordinates in the hyperbolic space,
        to the extrinsic coordinates in Minkowski space.
        """
        dimension = self.dimension
        point_extrinsic = np.zeros(dimension + 1, 'float')
        point_extrinsic[1: dimension + 1] = point_intrinsic[0: dimension]
        point_extrinsic[0] = np.sqrt(1. + np.dot(point_intrinsic,
                                                 point_intrinsic))
        return point_extrinsic

    def extrinsic_to_intrinsic_coords(self, point_extrinsic):
        """
        From the extrinsic coordinates in Minkowski space,
        to the extrinsic coordinates in Hyperbolic space.
        """
        return point_extrinsic[1:]

    def projection_to_tangent_space(self, vector, base_point):
        """
         Project the vector vector onto the tangent space at base_point
         T_{base_point}H
                = { w s.t. embedding_inner_product(base_point, w) = 0 }
        """
        assert self.belongs(base_point)

        inner_prod = self.embedding_metric.inner_product(base_point,
                                                         vector)
        sq_norm_base_point = self.embedding_metric.squared_norm(base_point)

        tangent_vec = vector - inner_prod * base_point / sq_norm_base_point
        return tangent_vec

    def random_uniform(self, max_norm=1):
        """
        Generate random elements on the hyperbolic space.
        """
        point = (np.random.random_sample(self.dimension) - .5) * max_norm
        point = self.intrinsic_to_extrinsic_coords(point)
        return point


class HyperbolicMetric(RiemannianMetric):

    def __init__(self, dimension):
        self.dimension = dimension
        self.signature = (dimension, 0, 0)
        self.embedding_metric = MinkowskiMetric(dimension + 1)

    def squared_norm(self, vector, base_point=None):
        """
        Squared norm associated to the Hyperbolic Metric.
        """
        sq_norm = self.embedding_metric.squared_norm(vector)
        return sq_norm

    def exp(self, tangent_vec, base_point, epsilon=EPSILON):
        """
        Compute the Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the metric obtained by
        embedding of the hyperbolic space in the Minkowski space.

        This gives a point on the hyperbolic space.

        :param base_point: a point on the hyperbolic space
        :param vector: vector
        :returns riem_exp: a point on the hyperbolic space
        """
        sq_norm_tangent_vec = self.embedding_metric.squared_norm(
                tangent_vec)
        norm_tangent_vec = math.sqrt(sq_norm_tangent_vec)

        if norm_tangent_vec < epsilon:
            coef_1 = (1. + COSH_TAYLOR_COEFFS[2] * norm_tangent_vec ** 2
                      + COSH_TAYLOR_COEFFS[4] * norm_tangent_vec ** 4
                      + COSH_TAYLOR_COEFFS[6] * norm_tangent_vec ** 6
                      + COSH_TAYLOR_COEFFS[8] * norm_tangent_vec ** 8)
            coef_2 = (1. + SINH_TAYLOR_COEFFS[3] * norm_tangent_vec ** 2
                      + SINH_TAYLOR_COEFFS[5] * norm_tangent_vec ** 4
                      + SINH_TAYLOR_COEFFS[7] * norm_tangent_vec ** 6
                      + SINH_TAYLOR_COEFFS[9] * norm_tangent_vec ** 8)
        else:
            coef_1 = np.cosh(norm_tangent_vec)
            coef_2 = np.sinh(norm_tangent_vec) / norm_tangent_vec

        riem_exp = coef_1 * base_point + coef_2 * tangent_vec

        return riem_exp

    def log(self, point, base_point, epsilon=EPSILON):
        """
        Compute the Riemannian logarithm at point base_point,
        of point wrt the metric obtained by
        embedding of the hyperbolic space in the Minkowski space.

        This gives a tangent vector at point base_point.

        :param base_point: point on the hyperbolic space
        :param point: point on the hyperbolic space
        :returns riem_log: tangent vector at base_point
        """
        angle = self.dist(base_point, point)
        if angle < epsilon:
            coef_1 = (1. + INV_SINH_TAYLOR_COEFFS[1] * angle ** 2
                      + INV_SINH_TAYLOR_COEFFS[3] * angle ** 4
                      + INV_SINH_TAYLOR_COEFFS[5] * angle ** 6
                      + INV_SINH_TAYLOR_COEFFS[7] * angle ** 8)
            coef_2 = (1. + INV_TANH_TAYLOR_COEFFS[1] * angle ** 2
                      + INV_TANH_TAYLOR_COEFFS[3] * angle ** 4
                      + INV_TANH_TAYLOR_COEFFS[5] * angle ** 6
                      + INV_TANH_TAYLOR_COEFFS[7] * angle ** 8)
        else:
            coef_1 = angle / np.sinh(angle)
            coef_2 = angle / np.tanh(angle)
        return coef_1 * point - coef_2 * base_point

    def dist(self, point_a, point_b):
        """
        Compute the distance induced on the hyperbolic
        space, from its embedding in the Minkowski space.
        """
        sq_norm_a = self.embedding_metric.squared_norm(point_a)
        sq_norm_b = self.embedding_metric.squared_norm(point_b)
        inner_prod = self.embedding_metric.inner_product(point_a, point_b)

        cosh_angle = - inner_prod / math.sqrt(sq_norm_a * sq_norm_b)

        if cosh_angle <= 1.:
            return 0.

        return np.arccosh(cosh_angle)
