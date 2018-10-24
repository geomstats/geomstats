"""
Parameterized manifold.
"""

import math

import numpy as np

import geomstats.backend as gs

from geomstats.embedded_manifold import EmbeddedManifold
from geomstats.euclidean_space import EuclideanMetric
from geomstats.euclidean_space import EuclideanSpace
from geomstats.riemannian_metric import RiemannianMetric


R2 = EuclideanSpace(dimension=2)
R3 = EuclideanSpace(dimension=3)


class DiscretizedCurvesSpace(EmbeddedManifold):
    """
    Space of parameterized curves sampled at points in embedding_manifold.
    """
    def __init__(self, embedding_manifold):
        super(DiscretizedCurvesSpace, self).__init__(
                dimension=math.inf,
                embedding_manifold=embedding_manifold)
        self.L2_metric = L2Metric(self.embedding_manifold)
        self.SRV_metric = SRVMetric(self.embedding_manifold)

    def belongs(self, point):
        return gs.all(self.embedding_manifold.belongs(point))


class L2Metric(RiemannianMetric):
    """
    L2 Riemannian metric on the space of parameterized curves or surfaces.
    """
    def __init__(self, embedding_manifold):
        super(L2Metric, self).__init__(
                dimension=embedding_manifold.dimension,
                signature=(embedding_manifold.dimension, 0, 0))
        self.embedding_metric = embedding_manifold.metric

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """
        Inner product between two tangent vectors at a base point.
        """
        assert tangent_vec_a.shape == tangent_vec_b.shape
        assert tangent_vec_a.shape == base_point.shape
        n_coord = base_point.shape[-1]
        n_sampling_points = base_point.shape[-2]
        shape_cumprod = np.cumprod(base_point.shape)
        tangent_vec_a = tangent_vec_a.reshape(shape_cumprod[-2], n_coord)
        tangent_vec_b = tangent_vec_b.reshape(shape_cumprod[-2], n_coord)

        inner_prod = self.embedding_metric.inner_product(
                tangent_vec_a, tangent_vec_b, base_point)
        inner_prod = inner_prod.reshape(base_point.shape[:-1])
        inner_prod = gs.sum(inner_prod, -1) / n_sampling_points

        return inner_prod

    def dist(self, point_a, point_b):
        """
        Geodesic distance between two parameterized manifolds
        sampled at points_a and points_b.
        """
        assert point_a.shape == point_b.shape
        shape_a = point_a.shape
        n_coord = shape_a[-1]
        n_sampling_points = shape_a[-2]
        shape_cumprod = np.cumprod(point_a.shape)
        point_a = point_a.reshape(shape_cumprod[-2], n_coord)
        point_b = point_b.reshape(shape_cumprod[-2], n_coord)

        dist = self.embedding_metric.dist(point_a, point_b)
        dist = dist.reshape(shape_a[:-1])
        dist = gs.sqrt(gs.sum(dist ** 2, -1) / n_sampling_points)

        return dist

    def exp(self, tangent_vec, base_point):
        """
        Riemannian exponential of a tangent vector wrt to a base point.
        """
        base_point = base_point.squeeze()
        mult_tangent_vecs = 0
        if tangent_vec.ndim > base_point.ndim:
            mult_tangent_vecs = 1

        point_shape = base_point.shape
        tangent_vec_shape = tangent_vec.shape

        n_coord = point_shape[-1]
        shape_cumprod = np.cumprod(point_shape)

        new_base_point = base_point.reshape(shape_cumprod[-2], n_coord)
        if mult_tangent_vecs == 1:
            new_shape = [tangent_vec.shape[0], shape_cumprod[-2], n_coord]
        else:
            new_shape = [shape_cumprod[-2], n_coord]
        new_tangent_vec = tangent_vec.reshape(new_shape)

        exp = self.embedding_metric.exp(new_tangent_vec, new_base_point)
        exp = exp.reshape(tangent_vec_shape)

        return exp

    def log(self, point, base_point):
        """
        Riemannian logarithm of a point wrt a base point.
        """
        assert point.shape == base_point.shape
        point_shape = point.shape
        n_coord = point_shape[-1]
        shape_cumprod = np.cumprod(point_shape)

        point = point.reshape(shape_cumprod[-2], n_coord)
        base_point = base_point.reshape(shape_cumprod[-2], n_coord)
        log = self.embedding_metric.log(point, base_point)
        log = log.reshape(point_shape)

        return log

    def geodesic(self, initial_point,
                 end_point=None, initial_tangent_vec=None):
        """
        Geodesic specified either by an initial point and an end point,
        either by an initial point and an initial tangent vector.
        """
        #TODO(alice): vectorize
        point_ndim = 2
        point_shape = initial_point.shape
        initial_point = gs.to_ndarray(initial_point,
                                      to_ndim=point_ndim+1)

        if end_point is None and initial_tangent_vec is None:
            raise ValueError('Specify an end point or an initial tangent '
                             'vector to define the geodesic.')
        if end_point is not None:
            end_point = gs.to_ndarray(end_point,
                                      to_ndim=point_ndim+1)
            shooting_tangent_vec = self.log(point=end_point,
                                            base_point=initial_point)
            if initial_tangent_vec is not None:
                assert gs.allclose(shooting_tangent_vec, initial_tangent_vec)
            initial_tangent_vec = shooting_tangent_vec
        initial_tangent_vec = gs.array(initial_tangent_vec)
        initial_tangent_vec = gs.to_ndarray(initial_tangent_vec,
                                            to_ndim=point_ndim+1)

        def point_on_geodesic(t):
            t = gs.cast(t, gs.float32)
            t = gs.to_ndarray(t, to_ndim=1)
            t = gs.to_ndarray(t, to_ndim=2, axis=1)
            new_initial_point = gs.to_ndarray(
                                          initial_point,
                                          to_ndim=point_ndim+1)
            new_initial_tangent_vec = gs.to_ndarray(
                                          initial_tangent_vec,
                                          to_ndim=point_ndim+1)

            tangent_vecs = gs.einsum('il,nkm->ikm', t, new_initial_tangent_vec)

            shape_point_at_time_t = gs.hstack([len(t), point_shape])
            point_at_time_t = gs.zeros(shape_point_at_time_t)
            for k in range(len(t)):
                point_at_time_t[k, :] = self.exp(
                        tangent_vec=tangent_vecs[k, :],
                        base_point=new_initial_point)
            return point_at_time_t

        return point_on_geodesic


class SRVMetric(RiemannianMetric):
    """
    Elastic metric defined using the Square Root Velocity Function
    (see Srivastava et al. 2011).
    """
    def __init__(self, embedding_manifold):
        super(SRVMetric, self).__init__(
                dimension=embedding_manifold.dimension,
                signature=(embedding_manifold.dimension, 0, 0))
        self.embedding_metric = embedding_manifold.metric
        self.L2_metric = L2Metric(embedding_manifold=embedding_manifold)

    def pointwise_inner_product(self, tangent_vec_a, tangent_vec_b,
                                base_point):
        """
        Compute the inner products of the components of a (series of)
        pair(s) of tangent vectors at (a) base point(s).
        """
        point_ndim = 2
        n_sampling_points = base_point.shape[-2]
        base_point = gs.to_ndarray(base_point, to_ndim=point_ndim + 1)
        tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=point_ndim + 1)
        tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=point_ndim + 1)

        n_tangent_vecs = tangent_vec_a.shape[0]
        inner_prod = gs.zeros([n_tangent_vecs, n_sampling_points])
        for k in range(n_tangent_vecs):
            inner_prod[k, :] = self.embedding_metric.inner_product(
                    tangent_vec_a[k, :], tangent_vec_b[k, :],
                    base_point[k, :]).squeeze()

        return inner_prod

    def pointwise_norm(self, tangent_vec, base_point):
        """
        Compute the norms of the components of a (series of)
        tangent vector(s) at (a) base point(s).
        """
        sq_norm = self.pointwise_inner_product(tangent_vec_a=tangent_vec,
                                               tangent_vec_b=tangent_vec,
                                               base_point=base_point)
        return gs.sqrt(sq_norm)

    def square_root_velocity(self, curve):
        """
        Compute the square root velocity representation of a curve.
        """
        n_coords = curve.shape[-1]
        n_sampling_points = curve.shape[-2]
        n_points_in_total = np.cumprod(curve.shape)[-2]
        srv_shape = gs.array(curve.shape)
        srv_shape[-2] -= 1

        curve = curve.reshape(n_points_in_total, n_coords)
        velocity = n_sampling_points * self.embedding_metric.log(
                point=curve[1:, :], base_point=curve[:-1, :])
        velocity_norm = self.embedding_metric.norm(velocity, curve[:-1, :])
        assert gs.all(velocity_norm != 0)
        srv = velocity / gs.sqrt(velocity_norm)
        index = gs.arange(n_points_in_total - 1)
        index_select = index[(index + 1) % n_sampling_points != 0]
        srv = srv[index_select, :].reshape(srv_shape)

        return srv

    def square_root_velocity_inverse(self, srv, origin):
        """
        Retreive a curve from its square root velocity representation
        and origin.
        """
        if not isinstance(self.embedding_metric, EuclideanMetric):
            raise AssertionError('The square root velocity inverse is only '
                                 'implemented for dicretized curves embedded '
                                 'in a Euclidean space.')
        if srv.ndim != origin.ndim:
            origin = gs.transpose(np.tile(origin, (1, 1, 1)), (1, 0, 2))
        srv_shape = srv.shape
        n_coords = srv_shape[-1]
        n_sampling_points = srv_shape[-2] + 1
        n_points_in_total = np.cumprod(srv_shape)[-2]

        srv = srv.reshape(n_points_in_total, n_coords)
        srv_norm = self.embedding_metric.norm(srv)
        delta_points = 1 / n_sampling_points * srv_norm * srv
        delta_points = delta_points.reshape(srv_shape)
        curve = np.concatenate((origin, delta_points), -2)
        curve = np.cumsum(curve, -2)

        return curve

    def exp(self, tangent_vec, base_point):
        """
        Riemannian exponential of a tangent vector wrt to a base point.
        """
        if not isinstance(self.embedding_metric, EuclideanMetric):
            raise AssertionError('The exponential map is only implemented '
                                 'for dicretized curves embedded in a '
                                 'Euclidean space.')
        point_ndim = 2
        n_sampling_points = base_point.shape[-2]

        base_point = gs.to_ndarray(base_point, to_ndim=point_ndim + 1)
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=point_ndim + 1)

        srv_base_point = self.square_root_velocity(base_point)

        tangent_vec_derivative = n_sampling_points * (
                tangent_vec[:, 1:, :] - tangent_vec[:, :-1, :])
        base_point_velocity = n_sampling_points * (
                base_point[:, 1:, :] - base_point[:, :-1, :])
        base_point_velocity_norm = self.pointwise_norm(base_point_velocity,
                                                       base_point[:, :-1, :])

        coef_1 = 1 / gs.sqrt(base_point_velocity_norm)
        coef_2 = - 1 / (2 * base_point_velocity_norm ** (5/2)) * \
            self.pointwise_inner_product(tangent_vec_derivative,
                                         base_point_velocity,
                                         base_point[:, :-1, :])
        srv_initial_derivative = gs.transpose(coef_1) * \
            gs.transpose(tangent_vec_derivative, (2, 1, 0)) + \
            gs.transpose(coef_2) * gs.transpose(base_point_velocity, (2, 1, 0))
        srv_initial_derivative = gs.transpose(srv_initial_derivative,
                                              (2, 1, 0))

        srv_end_point = self.L2_metric.exp(tangent_vec=srv_initial_derivative,
                                           base_point=srv_base_point)
        origin_end_point = self.embedding_metric.exp(
                tangent_vec=tangent_vec[:, 0, :],
                base_point=base_point[:, 0, :])
        end_point = self.square_root_velocity_inverse(srv_end_point,
                                                      origin_end_point)

        return end_point

    def log(self, point, base_point):
        """
        Riemannian logarithm of a point wrt a base point.
        """
        if not isinstance(self.embedding_metric, EuclideanMetric):
            raise AssertionError('The logarithm map is only implemented '
                                 'for dicretized curves embedded in a '
                                 'Euclidean space.')
        point_ndim = 2
        n_sampling_points = point.shape[-2]

        point = gs.to_ndarray(point, to_ndim=point_ndim + 1)
        base_point = gs.to_ndarray(base_point, to_ndim=point_ndim + 1)
        n_points = point.shape[0]

        srv_point = self.square_root_velocity(point)
        srv_base_point = self.square_root_velocity(base_point)
        base_point_velocity = n_sampling_points * (
                base_point[:, 1:, :] - base_point[:, :-1, :])
        base_point_velocity_norm = self.pointwise_norm(base_point_velocity,
                                                       base_point[:, :-1, :])

        coef_1 = gs.sqrt(base_point_velocity_norm)
        coef_2 = 1 / base_point_velocity_norm ** (3/2) * \
            self.pointwise_inner_product(srv_point - srv_base_point,
                                         base_point_velocity,
                                         base_point[:, :-1, :])
        log_derivative = gs.transpose(coef_1) * \
            gs.transpose(srv_point - srv_base_point, (2, 1, 0)) + \
            gs.transpose(coef_2) * gs.transpose(base_point_velocity, (2, 1, 0))
        log_derivative = gs.transpose(log_derivative, (2, 1, 0))
        log_0 = self.embedding_metric.log(point=point[:, 0, :],
                                          base_point=base_point[:, 0, :])
        log_0 = gs.transpose(np.tile(log_0, (1, 1, 1)), (1, 0, 2))

        log_cumsum = gs.hstack([gs.zeros((n_points, 1, 3)),
                                np.cumsum(log_derivative, -2)])
        log = log_0 + 1 / n_sampling_points * log_cumsum

        return log

    def geodesic(self, initial_point,
                 end_point=None, initial_tangent_vec=None):
        """
        Geodesic specified either by an initial point and an end point,
        either by an initial point and an initial tangent vector.
        """
        #TODO(alice): vectorize

        if not isinstance(self.embedding_metric, EuclideanMetric):
            raise AssertionError('The geodesics are only implemented for '
                                 'dicretized curves embedded in a '
                                 'Euclidean space.')
        point_ndim = 2
        point_shape = initial_point.shape
        initial_point = gs.to_ndarray(initial_point,
                                      to_ndim=point_ndim+1)

        if end_point is None and initial_tangent_vec is None:
            raise ValueError('Specify an end point or an initial tangent '
                             'vector to define the geodesic.')
        if end_point is not None:
            end_point = gs.to_ndarray(end_point,
                                      to_ndim=point_ndim+1)
            shooting_tangent_vec = self.log(point=end_point,
                                            base_point=initial_point)
            if initial_tangent_vec is not None:
                assert gs.allclose(shooting_tangent_vec, initial_tangent_vec)
            initial_tangent_vec = shooting_tangent_vec
        initial_tangent_vec = gs.array(initial_tangent_vec)
        initial_tangent_vec = gs.to_ndarray(initial_tangent_vec,
                                            to_ndim=point_ndim+1)

        def point_on_geodesic(t):
            t = gs.cast(t, gs.float32)
            t = gs.to_ndarray(t, to_ndim=1)
            t = gs.to_ndarray(t, to_ndim=2, axis=1)
            new_initial_point = gs.to_ndarray(
                                          initial_point,
                                          to_ndim=point_ndim+1)
            new_initial_tangent_vec = gs.to_ndarray(
                                          initial_tangent_vec,
                                          to_ndim=point_ndim+1)

            tangent_vecs = gs.einsum('il,nkm->ikm', t, new_initial_tangent_vec)

            shape_point_at_time_t = gs.hstack([len(t), point_shape])
            point_at_time_t = gs.zeros(shape_point_at_time_t)
            for k in range(len(t)):
                point_at_time_t[k, :] = self.exp(
                        tangent_vec=tangent_vecs[k, :],
                        base_point=new_initial_point)
            return point_at_time_t

        return point_on_geodesic

    def dist(self, point_a, point_b):
        """
        Geodesic distance between the curves sampled at points_a and points_b.
        """
        if not isinstance(self.embedding_metric, EuclideanMetric):
            raise AssertionError('The distance is only implemented for '
                                 'dicretized curves embedded in a '
                                 'Euclidean space.')
        assert point_a.shape == point_b.shape

        srvf_a = self.square_root_velocity(point_a)
        srvf_b = self.square_root_velocity(point_b)
        dist_origins = self.embedding_metric.dist(point_a[0, :], point_b[0, :])
        dist_srvs = self.L2_metric.dist(srvf_a, srvf_b)
        dist = gs.sqrt(dist_origins ** 2 + dist_srvs ** 2)

        return dist
