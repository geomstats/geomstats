"""
The n-dimensional hypersphere
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
                     - 1.0 / math.factorial(2), 0.,
                     + 1.0 / math.factorial(4), 0.,
                     - 1.0 / math.factorial(6), 0.,
                     + 1.0 / math.factorial(8), 0.]
INV_SIN_TAYLOR_COEFFS = [0., 1. / 6.,
                         0., 7. / 360.,
                         0., 31. / 15120.,
                         0., 127. / 604800.]
INV_TAN_TAYLOR_COEFFS = [0., - 1. / 3.,
                         0., - 1. / 45.,
                         0., - 2. / 945.,
                         0., -1. / 4725.]


class Hypersphere(EmbeddedManifold):
    """
    Class for the n-dimensional hypersphere
    embedded in the (n+1)-dimensional Euclidean space.

    By default, points are parameterized by their extrinsic (n+1)-coordinates.
    """

    def __init__(self, dimension):
        assert isinstance(dimension, int) and dimension > 0
        super(Hypersphere, self).__init__(
                dimension=dimension,
                embedding_manifold=EuclideanSpace(dimension+1))
        self.embedding_metric = self.embedding_manifold.metric
        self.metric = HypersphereMetric(dimension)

    def belongs(self, point, tolerance=TOLERANCE):
        """
        Evaluate if a point belongs to the Hypersphere,
        i.e. evaluate if its squared norm in the Euclidean space is 1.
        """
        point = gs.asarray(point)
        point_dim = point.shape[-1]
        if point_dim != self.dimension + 1:
            if point_dim is self.dimension:
                logging.warning(
                    'Use the extrinsic coordinates to '
                    'represent points on the hypersphere.')
            return gs.array([[False]])
        sq_norm = self.embedding_metric.squared_norm(point)
        diff = gs.abs(sq_norm - 1)
        return gs.less_equal(diff, tolerance)

    def regularize(self, point):
        """
        Regularize a point to the canonical representation
        chosen for the Hypersphere, to avoid numerical issues.
        """
        assert gs.all(self.belongs(point))

        return self.projection(point)

    def projection(self, point):
        """
        Project a point on the Hypersphere.
        """
        point = gs.to_ndarray(point, to_ndim=2)

        norm = self.embedding_metric.norm(point)
        projected_point = point / norm

        return projected_point

    def projection_to_tangent_space(self, vector, base_point):
        """
        Project a vector in Euclidean space
        on the tangent space of the Hypersphere at a base point.
        """
        vector = gs.to_ndarray(vector, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)

        sq_norm = self.embedding_metric.squared_norm(base_point)
        inner_prod = self.embedding_metric.inner_product(base_point, vector)
        coef = inner_prod / sq_norm
        tangent_vec = vector - gs.einsum('ni,nj->nj', coef, base_point)

        return tangent_vec

    def intrinsic_to_extrinsic_coords(self, point_intrinsic):
        """
        Convert from the intrinsic coordinates in the Hypersphere,
        to the extrinsic coordinates in Euclidean space.
        """
        point_intrinsic = gs.to_ndarray(point_intrinsic, to_ndim=2)

        coord_0 = gs.sqrt(1. - gs.linalg.norm(point_intrinsic, axis=-1) ** 2)
        coord_0 = gs.to_ndarray(coord_0, to_ndim=2, axis=-1)

        point_extrinsic = gs.concatenate([coord_0, point_intrinsic], axis=-1)

        return point_extrinsic

    def extrinsic_to_intrinsic_coords(self, point_extrinsic):
        """
        Convert from the extrinsic coordinates in Euclidean space,
        to some intrinsic coordinates in Hypersphere.
        """
        point_extrinsic = gs.to_ndarray(point_extrinsic, to_ndim=2)

        point_intrinsic = point_extrinsic[:, 1:]

        return point_intrinsic

    def random_uniform(self, n_samples=1, bound=0.5):
        """
        Sample in the Hypersphere with the uniform distribution.
        """
        size = (n_samples, self.dimension)

        if bound is None:
            spherical_coord = gs.random.rand(*size) * gs.pi
            spherical_coord[:, -1] *= 2

            point = gs.zeros((n_samples, self.dimension+1))
            for i in range(self.dimension):
                point[:, i] = (
                        gs.prod(gs.sin(spherical_coord[:, :i]), axis=1)
                        * gs.cos(spherical_coord[:, i]))
            point[:, -1] = gs.prod(gs.sin(spherical_coord), axis=1)

        else:
            assert bound <= 0.5
            point = bound * (2 * gs.random.rand(*size) - 1)
            point = self.intrinsic_to_extrinsic_coords(point)

        return point

    def random_von_mises_fisher(self, kappa=10, n_samples=1):
        """
        Sample in the 2-sphere with the von Mises distribution
        centered in the north pole.
        """
        if self.dimension != 2:
            raise NotImplementedError(
                    'Sampling from the von Mises Fisher distribution'
                    'is only implemented in dimension 2.')
        angle = 2. * gs.pi * gs.random.rand(n_samples)
        angle = gs.to_ndarray(angle, to_ndim=2, axis=1)
        unit_vector = gs.hstack((gs.cos(angle), gs.sin(angle)))
        scalar = gs.random.rand(n_samples)

        coord_z = 1. + 1. / kappa * gs.log(
                      scalar + (1. - scalar) * gs.exp(-2. * kappa))
        coord_z = gs.to_ndarray(coord_z, to_ndim=2, axis=1)

        coord_xy = gs.sqrt(1. - coord_z**2) * unit_vector

        point = gs.hstack((coord_xy, coord_z))

        return point


class HypersphereMetric(RiemannianMetric):

    def __init__(self, dimension):
        super(HypersphereMetric, self).__init__(
                dimension=dimension,
                signature=(dimension, 0, 0))
        self.embedding_metric = EuclideanMetric(dimension + 1)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """
        Inner product.
        """
        inner_prod = self.embedding_metric.inner_product(
                tangent_vec_a, tangent_vec_b, base_point)

        return inner_prod

    def squared_norm(self, vector, base_point=None):
        """
        Squared norm of a vector associated to the inner product
        at the tangent space at a base point.
        """
        sq_norm = self.embedding_metric.squared_norm(vector)
        return sq_norm

    def exp(self, tangent_vec, base_point):
        """
        Riemannian exponential of a tangent vector wrt to a base point.
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        # TODO(johmathe): Evaluate the bias introduced by this variable
        norm_tangent_vec = self.embedding_metric.norm(tangent_vec) + EPSILON
        coef_1 = gs.cos(norm_tangent_vec)
        coef_2 = gs.sin(norm_tangent_vec) / norm_tangent_vec
        exp = (gs.einsum('ni,nj->nj', coef_1, base_point)
               + gs.einsum('ni,nj->nj', coef_2, tangent_vec))

        return exp

    def log(self, point, base_point):
        """
        Riemannian logarithm of a point wrt a base point.
        """
        point = gs.to_ndarray(point, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)

        norm_base_point = self.embedding_metric.norm(base_point)
        norm_point = self.embedding_metric.norm(point)
        inner_prod = self.embedding_metric.inner_product(base_point, point)
        cos_angle = inner_prod / (norm_base_point * norm_point)
        cos_angle = gs.clip(cos_angle, -1., 1.)

        angle = gs.arccos(cos_angle)
        angle = gs.to_ndarray(angle, to_ndim=1)
        angle = gs.to_ndarray(angle, to_ndim=2, axis=1)

        mask_0 = gs.isclose(angle, 0.)
        mask_else = gs.equal(mask_0, gs.array(False))

        mask_0_float = gs.cast(mask_0, gs.float32)
        mask_else_float = gs.cast(mask_else, gs.float32)

        coef_1 = gs.zeros_like(angle)
        coef_2 = gs.zeros_like(angle)

        coef_1 += mask_0_float * (
           1. + INV_SIN_TAYLOR_COEFFS[1] * angle ** 2
           + INV_SIN_TAYLOR_COEFFS[3] * angle ** 4
           + INV_SIN_TAYLOR_COEFFS[5] * angle ** 6
           + INV_SIN_TAYLOR_COEFFS[7] * angle ** 8)
        coef_2 += mask_0_float * (
           1. + INV_TAN_TAYLOR_COEFFS[1] * angle ** 2
           + INV_TAN_TAYLOR_COEFFS[3] * angle ** 4
           + INV_TAN_TAYLOR_COEFFS[5] * angle ** 6
           + INV_TAN_TAYLOR_COEFFS[7] * angle ** 8)

        # This avoids division by 0.
        angle += mask_0_float * 1.

        coef_1 += mask_else_float * angle / gs.sin(angle)
        coef_2 += mask_else_float * angle / gs.tan(angle)

        log = (gs.einsum('ni,nj->nj', coef_1, point)
               - gs.einsum('ni,nj->nj', coef_2, base_point))

        mask_same_values = gs.isclose(point, base_point)

        mask_else = gs.equal(mask_same_values, gs.array(False))
        mask_else_float = gs.cast(mask_else, gs.float32)
        mask_else_float = gs.to_ndarray(mask_else_float, to_ndim=1)
        mask_else_float = gs.to_ndarray(mask_else_float, to_ndim=2)
        mask_not_same_points = gs.sum(mask_else_float, axis=1)
        mask_same_points = gs.isclose(mask_not_same_points, 0.)
        mask_same_points = gs.cast(mask_same_points, gs.float32)
        mask_same_points = gs.to_ndarray(mask_same_points, to_ndim=2, axis=1)

        mask_same_points_float = gs.cast(mask_same_points, gs.float32)

        log -= mask_same_points_float * log

        return log

    def dist(self, point_a, point_b):
        """
        Geodesic distance between two points.
        """
        norm_a = self.embedding_metric.norm(point_a)
        norm_b = self.embedding_metric.norm(point_b)
        inner_prod = self.embedding_metric.inner_product(point_a, point_b)

        cos_angle = inner_prod / (norm_a * norm_b)
        cos_angle = gs.clip(cos_angle, -1, 1)

        dist = gs.arccos(cos_angle)

        return dist
