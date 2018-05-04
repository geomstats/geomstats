"""
Computations on the n-dimensional sphere
embedded in the (n+1)-dimensional Euclidean space.
"""

import logging
import math

import geomstats.backend as gs
from geomstats.embedded_manifold import EmbeddedManifold
from geomstats.euclidean_space import EuclideanMetric
from geomstats.euclidean_space import EuclideanSpace
from geomstats.riemannian_metric import RiemannianMetric

TOLERANCE = 1e-6
EPSILON = 1e-8

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


class Hypersphere(EmbeddedManifold):
    """Hypersphere embedded in Euclidean space."""

    def __init__(self, dimension):
        assert isinstance(dimension, int) and dimension > 0
        super(Hypersphere, self).__init__(
                dimension=dimension,
                embedding_manifold=EuclideanSpace(dimension+1))
        self.embedding_metric = self.embedding_manifold.metric
        self.metric = HypersphereMetric(dimension)

    def belongs(self, point, tolerance=TOLERANCE):
        """
        By definition, a point on the Hypersphere has squared norm 1
        in the embedding Euclidean space.
        Note: point must be given in extrinsic coordinates.
        """
        point = gs.to_ndarray(point, to_ndim=2)

        _, point_dim = point.shape
        if point_dim is not self.dimension + 1:
            if point_dim is self.dimension:
                logging.warning('Use the extrinsic coordinates to '
                                'represent points on the hypersphere.')
            return False
        sq_norm = self.embedding_metric.squared_norm(point)
        diff = gs.abs(sq_norm - 1)

        return diff < tolerance

    def projection_to_tangent_space(self, vector, base_point):
        """
        Project the vector vector onto the tangent space:
        T_{base_point} S = {w | scal(w, base_point) = 0}
        """
        gs.Assert(gs.all(self.belongs(base_point)))

        sq_norm = self.embedding_metric.squared_norm(base_point)
        inner_prod = self.embedding_metric.inner_product(base_point, vector)
        tangent_vec = (vector - inner_prod / sq_norm * base_point)

        return tangent_vec

    def intrinsic_to_extrinsic_coords(self, point_intrinsic):
        """
        From some intrinsic coordinates in the Hypersphere,
        to the extrinsic coordinates in Euclidean space.
        """
        point_intrinsic = gs.to_ndarray(point_intrinsic, to_ndim=2)

        coord_0 = gs.sqrt(1. - gs.linalg.norm(point_intrinsic, axis=1) ** 2)
        coord_0 = gs.to_ndarray(coord_0, to_ndim=2, axis=1)

        point_extrinsic = gs.hstack([coord_0, point_intrinsic])

        gs.Assert(gs.all(self.belongs(point_extrinsic)))

        assert(point_extrinsic.ndim == 2)
        return point_extrinsic

    def extrinsic_to_intrinsic_coords(self, point_extrinsic):
        """
        From the extrinsic coordinates in Euclidean space,
        to some intrinsic coordinates in Hypersphere.
        """
        point_extrinsic = gs.to_ndarray(point_extrinsic, to_ndim=2)
        assert gs.all(self.belongs(point_extrinsic))

        point_intrinsic = point_extrinsic[:, 1:]
        assert point_intrinsic.ndim == 2
        return point_intrinsic

    def random_uniform(self, n_samples=1, max_norm=1):
        """
        Generate random elements on the Hypersphere.
        """
        point = ((gs.random.rand(n_samples, self.dimension) - .5)
                 * max_norm)
        point = self.intrinsic_to_extrinsic_coords(point)
        assert(gs.all(self.belongs(point)))

        assert(point.ndim == 2)
        return point


class HypersphereMetric(RiemannianMetric):

    def __init__(self, dimension):
        self.dimension = dimension
        self.signature = (dimension, 0, 0)
        self.embedding_metric = EuclideanMetric(dimension + 1)

    def squared_norm(self, vector, base_point=None):
        """
        Squared norm associated to the Hyperbolic Metric.
        """
        sq_norm = self.embedding_metric.squared_norm(vector)
        return sq_norm

    def projection_to_tangent_space(self, vector, base_point):
        """
        Project the vector vector onto the tangent space:
        T_{base_point} S = {w | scal(w, base_point) = 0}
        """
        # TODO(nina): define HypersphereMetric inside Hypersphere
        # to avoid copy-pasting this code?

        sq_norm = self.embedding_metric.squared_norm(base_point)
        inner_prod = self.embedding_metric.inner_product(base_point, vector)
        tangent_vec = (vector - inner_prod / sq_norm * base_point)

        return tangent_vec

    def exp(self, tangent_vec, base_point):
        """
        Compute the Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the metric obtained by
        embedding of the n-dimensional sphere
        in the (n+1)-dimensional euclidean space.

        This gives a point on the n-dimensional sphere.

        :param base_point: a point on the n-dimensional sphere
        :param vector: (n+1)-dimensional vector
        :return exp: a point on the n-dimensional sphere
        """
        # TODO(johmathe): Evaluate the bias introduced by this variable
        norm_tangent_vec = self.embedding_metric.norm(tangent_vec) + EPSILON
        coef_1 = gs.cos(norm_tangent_vec)
        coef_2 = gs.sin(norm_tangent_vec) / norm_tangent_vec

        exp = coef_1 * base_point + coef_2 * tangent_vec

        return exp

    def log(self, point, base_point):
        """
        Compute the Riemannian logarithm at point base_point,
        of point wrt the metric obtained by
        embedding of the n-dimensional sphere
        in the (n+1)-dimensional euclidean space.

        This gives a tangent vector at point base_point.

        :param base_point: point on the n-dimensional sphere
        :param point: point on the n-dimensional sphere
        :return log: tangent vector at base_point
        """
        norm_base_point = self.embedding_metric.norm(base_point)
        norm_point = self.embedding_metric.norm(point)
        inner_prod = self.embedding_metric.inner_product(base_point, point)
        cos_angle = inner_prod / (norm_base_point * norm_point)

        mask_greater_1 = cos_angle >= 1
        mask_else_1 = ~mask_greater_1

        angle = gs.zeros_like(cos_angle)
        angle[mask_greater_1] = 0.
        angle[mask_else_1] = gs.arccos(cos_angle[mask_else_1])

        mask_0 = gs.isclose(angle, 0)
        mask_else_0 = ~mask_0

        coef_1 = gs.zeros_like(angle)
        coef_2 = gs.zeros_like(angle)

        coef_1[mask_0] = (
                      1. + INV_SIN_TAYLOR_COEFFS[1] * angle[mask_0] ** 2
                      + INV_SIN_TAYLOR_COEFFS[3] * angle[mask_0] ** 4
                      + INV_SIN_TAYLOR_COEFFS[5] * angle[mask_0] ** 6
                      + INV_SIN_TAYLOR_COEFFS[7] * angle[mask_0] ** 8)
        coef_2[mask_0] = (
                      1. + INV_TAN_TAYLOR_COEFFS[1] * angle[mask_0] ** 2
                      + INV_TAN_TAYLOR_COEFFS[3] * angle[mask_0] ** 4
                      + INV_TAN_TAYLOR_COEFFS[5] * angle[mask_0] ** 6
                      + INV_TAN_TAYLOR_COEFFS[7] * angle[mask_0] ** 8)

        coef_1[mask_else_0] = angle[mask_else_0] / gs.sin(angle[mask_else_0])
        coef_2[mask_else_0] = angle[mask_else_0] / gs.tan(angle[mask_else_0])

        log = coef_1 * point - coef_2 * base_point

        return log

    def dist(self, point_a, point_b):
        """
        Compute the Riemannian distance between points
        point_a and point_b.
        """
        # TODO(nina): case gs.dot(unit_vec, unit_vec) != 1
        if gs.all(point_a == point_b):
            return 0.

        point_a = gs.to_ndarray(point_a, to_ndim=2)
        point_b = gs.to_ndarray(point_b, to_ndim=2)

        n_points_a, _ = point_a.shape
        n_points_b, _ = point_b.shape

        assert (n_points_a == n_points_b
                or n_points_a == 1
                or n_points_b == 1)

        n_dists = gs.maximum(n_points_a, n_points_b)
        dist = gs.zeros((n_dists, 1))

        norm_a = self.embedding_metric.norm(point_a)
        norm_b = self.embedding_metric.norm(point_b)
        inner_prod = self.embedding_metric.inner_product(point_a, point_b)

        cos_angle = inner_prod / (norm_a * norm_b)
        mask_cos_greater_1 = gs.greater_equal(cos_angle, 1.)
        mask_cos_less_minus_1 = gs.less_equal(cos_angle, -1.)
        mask_else = ~mask_cos_greater_1 & ~mask_cos_less_minus_1

        dist[mask_cos_greater_1] = 0.
        dist[mask_cos_less_minus_1] = gs.pi
        dist[mask_else] = gs.arccos(cos_angle[mask_else])

        return dist
