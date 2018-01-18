"""
Computations on the n-dimensional sphere
embedded in the (n+1)-dimensional Euclidean space.
"""

import numpy as np
import math

import EuclideanSpace
import EuclideanMetric

import Manifold
import RiemannianMetric


EPSILON = 1e-6
TOLERANCE = 1e-12

SIN_TAYLOR_COEFFS = [0., 1.,
                     0., - 1 / math.factorial(3),
                     0., + 1 / math.factorial(5),
                     0., - 1 / math.factorial(7),
                     0., + 1 / math.factorial(9)]
COS_TAYLOR_COEFFS = [1., 0.,
                     - 1 / math.factorial(2), 0.,
                     + 1 / math.factorial(4), 0.,
                     - 1 / math.factorial(6), 0.,
                     + 1 / math.factorial(8), 0.]
INV_SIN_TAYLOR_COEFFS = [0., 1. / 6.,
                         0., 7. / 360.,
                         0., 31. / 15120.,
                         0., 127. / 604800.]
INV_TAN_TAYLOR_COEFFS = [0., - 1. / 3.,
                         0., - 1. / 45.,
                         0., - 2. / 945.,
                         0., -1. / 4725.]


class HypersphereMetric(RiemannianMetric):

    def __init__(self):
        self.embedding_metric = EuclideanMetric()

    def riemannian_exp(self, ref_point, vector, epsilon=EPSILON):
        """
        Compute the Riemannian exponential at point ref_point
        of tangent vector tangent_vec wrt the metric obtained by
        embedding of the n-dimensional sphere
        in the (n+1)-dimensional euclidean space.

        This gives a point on the n-dimensional sphere.

        :param ref_point: a point on the n-dimensional sphere
        :param vector: (n+1)-dimensional vector
        :return riem_exp: a point on the n-dimensional sphere
        """
        tangent_vec = self.projection_to_tangent_space(ref_point, vector)
        norm_tangent_vec = self.embedding_metric.riemannian_norm(tangent_vec)

        if norm_tangent_vec < epsilon:
            coef_1 = (1. + COS_TAYLOR_COEFFS[2] * norm_tangent_vec ** 2
                      + COS_TAYLOR_COEFFS[4] * norm_tangent_vec ** 4
                      + COS_TAYLOR_COEFFS[6] * norm_tangent_vec ** 6
                      + COS_TAYLOR_COEFFS[8] * norm_tangent_vec ** 8)
            coef_2 = (1. + SIN_TAYLOR_COEFFS[3] * norm_tangent_vec ** 2
                      + SIN_TAYLOR_COEFFS[5] * norm_tangent_vec ** 4
                      + SIN_TAYLOR_COEFFS[7] * norm_tangent_vec ** 6
                      + SIN_TAYLOR_COEFFS[9] * norm_tangent_vec ** 8)
        else:
            coef_1 = np.cos(norm_tangent_vec)
            coef_2 = np.sin(norm_tangent_vec) / norm_tangent_vec

        riem_exp = coef_1 * ref_point + coef_2 * tangent_vec

        return riem_exp

    def riemannian_log(self, ref_point, point, epsilon=EPSILON):
        """
        Compute the Riemannian logarithm at point ref_point,
        of point wrt the metric obtained by
        embedding of the n-dimensional sphere
        in the (n+1)-dimensional euclidean space.

        This gives a tangent vector at point ref_point.

        :param ref_point: point on the n-dimensional sphere
        :param point: point on the n-dimensional sphere
        :return riem_log: tangent vector at ref_point
        """
        norm_ref_point = self.embedding_metric.riemannian_norm(ref_point)
        norm_point = self.embedding_metric.riemannian_norm(point)
        inner_prod = self.embedding_metric.riemannian_inner_product(ref_point,
                                                                    point)
        cos_angle = inner_prod / (norm_ref_point * norm_point)
        if cos_angle >= 1.:
            angle = 0.
        else:
            angle = np.arccos(cos_angle)

        if angle < epsilon:
            coef_1 = (1. + INV_SIN_TAYLOR_COEFFS[1] * angle ** 2
                      + INV_SIN_TAYLOR_COEFFS[3] * angle ** 4
                      + INV_SIN_TAYLOR_COEFFS[5] * angle ** 6
                      + INV_SIN_TAYLOR_COEFFS[7] * angle ** 8)
            coef_2 = (1. + INV_TAN_TAYLOR_COEFFS[1] * angle ** 2
                      + INV_TAN_TAYLOR_COEFFS[3] * angle ** 4
                      + INV_TAN_TAYLOR_COEFFS[5] * angle ** 6
                      + INV_TAN_TAYLOR_COEFFS[7] * angle ** 8)
        else:
            coef_1 = angle / np.sin(angle)
            coef_2 = angle / np.tan(angle)

        riem_log = coef_1 * point - coef_2 * ref_point

        return riem_log

    def riemannian_dist(self, point_a, point_b):
        """
        Compute the Riemannian distance between points
        point_a and point_b.
        """
        # TODO(nina): case np.dot(unit_vec, unit_vec) != 1
        if np.all(point_a == point_b):
            return 0.

        norm_a = self.embedding_metric.riemannian_norm(point_a)
        norm_b = self.embedding_metric.riemannian_norm(point_b)
        inner_prod = self.embedding_metric.riemannian_inner_product(point_a,
                                                                    point_b)

        cos_angle = inner_prod / (norm_a * norm_b)
        if cos_angle >= 1.:
            riem_dist = 0.
        elif cos_angle <= -1.:
            riem_dist = np.pi
        else:
            riem_dist = np.arccos(cos_angle)

        return riem_dist


class HyperSphere(Manifold):
    """Hypersphere embedded in Euclidean space."""

    def __init__(self, dimension):
        Manifold.__init__(dimension)
        self.riemannian_metric = HypersphereMetric()
        self.embedding_metric = EuclideanMetric()

    def belongs(self, point, tolerance=TOLERANCE):
        """
        By definition, a point on the hypersphere has squared norm 1
        in the embedding Euclidean space.
        Note: point must be given in extrinsic coordinates.
        """
        sq_norm = self.embedding_metric.riemannian_squared_norm(point)
        assert abs(sq_norm - 1) < tolerance

    def projection_to_tangent_space(self, ref_point, vector):
        """
        Project the vector vector onto the tangent space:
        T_{ref_point} S = {w | scal(w, ref_point) = 0}
        """
        assert self.belongs(ref_point)

        sq_norm = self.embedding_metric.riemannian_squared_norm(ref_point)
        inner_prod = self.embedding_metric.riemannian_inner_product(ref_point,
                                                                    vector)
        tangent_vec = (vector - inner_prod / sq_norm * ref_point)

        return tangent_vec
