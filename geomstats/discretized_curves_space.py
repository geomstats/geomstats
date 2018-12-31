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


def get_mask_i_float(i, n):
    range_n = gs.arange(n)
    i_float = gs.cast(gs.array([i]), gs.int32)[0]
    mask_i = gs.equal(range_n, i_float)
    mask_i_float = gs.cast(mask_i, gs.float32)
    return mask_i_float


class DiscretizedCurvesSpace(EmbeddedManifold):
    """
    Space of discretized curves sampled at points in embedding_manifold.
    """
    def __init__(self, embedding_manifold):
        super(DiscretizedCurvesSpace, self).__init__(
                dimension=math.inf,
                embedding_manifold=embedding_manifold)
        self.l2_metric = L2Metric(self.embedding_manifold)
        self.square_root_velocity_metric = SRVMetric(self.embedding_manifold)

    def belongs(self, point):
        belongs = gs.all(self.embedding_manifold.belongs(point))
        belongs = gs.to_ndarray(belongs, to_ndim=1)
        belongs = gs.to_ndarray(belongs, to_ndim=2, axis=1)
        return belongs


class L2Metric(RiemannianMetric):
    """
    L2 Riemannian metric on the space of discretized curves.
    """
    def __init__(self, embedding_manifold):
        super(L2Metric, self).__init__(
                dimension=math.inf,
                signature=(math.inf, 0, 0))
        self.embedding_manifold = embedding_manifold
        self.embedding_metric = embedding_manifold.metric

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_curve):
        """
        Inner product between two tangent vectors at a base curve.
        """
        assert tangent_vec_a.shape == tangent_vec_b.shape
        assert tangent_vec_a.shape == base_curve.shape
        tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=3)
        tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=3)
        base_curve = gs.to_ndarray(base_curve, to_ndim=3)

        n_curves, n_sampling_points, n_coords = tangent_vec_a.shape

        new_dim = n_curves * n_sampling_points
        tangent_vec_a = gs.reshape(tangent_vec_a, (new_dim, n_coords))
        tangent_vec_b = gs.reshape(tangent_vec_b, (new_dim, n_coords))
        base_curve = gs.reshape(base_curve, (new_dim, n_coords))

        inner_prod = self.embedding_metric.inner_product(
                tangent_vec_a, tangent_vec_b, base_curve)
        inner_prod = gs.reshape(inner_prod, (n_curves, n_sampling_points))
        inner_prod = gs.sum(inner_prod, -1)

        n_sampling_points_float = gs.array(n_sampling_points)
        n_sampling_points_float = gs.cast(n_sampling_points_float, gs.float32)
        inner_prod = inner_prod / n_sampling_points_float
        inner_prod = gs.to_ndarray(inner_prod, to_ndim=1)
        inner_prod = gs.to_ndarray(inner_prod, to_ndim=2, axis=1)

        return inner_prod

    def dist(self, curve_a, curve_b):
        """
        Geodesic distance between two discretized curves.
        """
        assert curve_a.shape == curve_b.shape
        curve_a = gs.to_ndarray(curve_a, to_ndim=3)
        curve_b = gs.to_ndarray(curve_b, to_ndim=3)

        n_curves, n_sampling_points, n_coords = curve_a.shape

        curve_a = gs.reshape(curve_a, (n_curves * n_sampling_points, n_coords))
        curve_b = gs.reshape(curve_b, (n_curves * n_sampling_points, n_coords))

        dist = self.embedding_metric.dist(curve_a, curve_b)
        dist = gs.reshape(dist, (n_curves, n_sampling_points))
        n_sampling_points_float = gs.array(n_sampling_points)
        n_sampling_points_float = gs.cast(n_sampling_points_float, gs.float32)
        dist = gs.sqrt(gs.sum(dist ** 2, -1) / n_sampling_points_float)
        dist = gs.to_ndarray(dist, to_ndim=1)
        dist = gs.to_ndarray(dist, to_ndim=2, axis=1)

        return dist

    def exp(self, tangent_vec, base_curve):
        """
        Riemannian exponential of a tangent vector wrt to a base curve.
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
        base_curve = gs.to_ndarray(base_curve, to_ndim=3)

        n_curves, n_sampling_points, n_coords = base_curve.shape
        n_tangent_vecs = tangent_vec.shape[0]

        new_dim = n_curves * n_sampling_points
        new_base_curve = gs.reshape(base_curve, (new_dim, n_coords))
        new_tangent_vec = gs.reshape(tangent_vec, (new_dim, n_coords))

        exp = self.embedding_metric.exp(new_tangent_vec, new_base_curve)
        exp = gs.reshape(exp, (n_tangent_vecs, n_sampling_points, n_coords))
        exp = gs.squeeze(exp)

        return exp

    def log(self, curve, base_curve):
        """
        Riemannian logarithm of a curve wrt a base curve.
        """
        assert curve.shape == base_curve.shape
        curve = gs.to_ndarray(curve, to_ndim=3)
        base_curve = gs.to_ndarray(base_curve, to_ndim=3)

        n_curves, n_sampling_points, n_coords = curve.shape

        curve = gs.reshape(
            curve, (n_curves * n_sampling_points, n_coords))
        base_curve = gs.reshape(
            base_curve, (n_curves * n_sampling_points, n_coords))
        log = self.embedding_metric.log(curve, base_curve)
        log = gs.reshape(log, (n_curves, n_sampling_points, n_coords))
        log = gs.squeeze(log)

        return log

    def geodesic(self, initial_curve,
                 end_curve=None, initial_tangent_vec=None):
        """
        Geodesic specified either by an initial point and an end point,
        either by an initial point and an initial tangent vector.
        """
        curve_ndim = 2
        initial_curve = gs.to_ndarray(initial_curve,
                                      to_ndim=curve_ndim+1)

        if end_curve is None and initial_tangent_vec is None:
            raise ValueError('Specify an end curve or an initial tangent '
                             'vector to define the geodesic.')
        if end_curve is not None:
            end_curve = gs.to_ndarray(end_curve,
                                      to_ndim=curve_ndim+1)
            shooting_tangent_vec = self.log(curve=end_curve,
                                            base_curve=initial_curve)
            if initial_tangent_vec is not None:
                assert gs.allclose(shooting_tangent_vec, initial_tangent_vec)
            initial_tangent_vec = shooting_tangent_vec
        initial_tangent_vec = gs.array(initial_tangent_vec)
        initial_tangent_vec = gs.to_ndarray(initial_tangent_vec,
                                            to_ndim=curve_ndim+1)

        def curve_on_geodesic(t):
            t = gs.cast(t, gs.float32)
            t = gs.to_ndarray(t, to_ndim=1)
            t = gs.to_ndarray(t, to_ndim=2, axis=1)
            new_initial_curve = gs.to_ndarray(initial_curve,
                                              to_ndim=curve_ndim+1)
            new_initial_tangent_vec = gs.to_ndarray(initial_tangent_vec,
                                                    to_ndim=curve_ndim+1)

            tangent_vecs = gs.einsum('il,nkm->ikm', t, new_initial_tangent_vec)

            def point_on_curve(tangent_vec):
                assert gs.ndim(tangent_vec) >= 2
                exp = self.exp(
                    tangent_vec=tangent_vec,
                    base_curve=new_initial_curve)
                return exp

            curve_at_time_t = gs.vectorize(
                tangent_vecs,
                point_on_curve,
                signature='(i,j)->(i,j)')

            return curve_at_time_t

        return curve_on_geodesic


class SRVMetric(RiemannianMetric):
    """
    Elastic metric defined using the Square Root Velocity Function
    (see Srivastava et al. 2011).
    """
    def __init__(self, embedding_manifold):
        super(SRVMetric, self).__init__(
                dimension=math.inf,
                signature=(math.inf, 0, 0))
        self.embedding_metric = embedding_manifold.metric
        self.l2_metric = L2Metric(embedding_manifold=embedding_manifold)

    def pointwise_inner_product(self, tangent_vec_a, tangent_vec_b,
                                base_curve):
        """
        Compute the inner products of the components of a (series of)
        pair(s) of tangent vectors at (a) base curve(s).
        """
        base_curve = gs.to_ndarray(base_curve, to_ndim=3)
        tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=3)
        tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=3)

        n_tangent_vecs = tangent_vec_a.shape[0]
        n_sampling_points = tangent_vec_a.shape[1]
        inner_prod = gs.zeros([n_tangent_vecs, n_sampling_points])

        def inner_prod_aux(vec_a, vec_b, curve):
            inner_prod = self.embedding_metric.inner_product(
                vec_a, vec_b, curve)
            return gs.squeeze(inner_prod)

        inner_prod = gs.vectorize(
            (tangent_vec_a, tangent_vec_b, base_curve),
            lambda x, y, z: inner_prod_aux(x, y, z),
            dtype=gs.float32,
            multiple_args=True,
            signature='(i,j),(i,j),(i,j)->(i)')

        return inner_prod

    def pointwise_norm(self, tangent_vec, base_curve):
        """
        Compute the norms of the components of a (series of)
        tangent vector(s) at (a) base curve(s).
        """
        sq_norm = self.pointwise_inner_product(
                tangent_vec_a=tangent_vec,
                tangent_vec_b=tangent_vec,
                base_curve=base_curve)
        return gs.sqrt(sq_norm)

    def square_root_velocity(self, curve):
        """
        Compute the square root velocity representation of a curve.

        The velocity is computed using the log map. The case of several curves
        is handled through vectorization. In that case, an index selection
        procedure allows to get rid of the log between the end point of
        curve[k, :, :] and the starting point of curve[k + 1, :, :].
        """
        curve = gs.to_ndarray(curve, to_ndim=3)
        n_curves, n_sampling_points, n_coords = curve.shape
        srv_shape = (n_curves, n_sampling_points-1, n_coords)

        curve = gs.reshape(curve, (n_curves * n_sampling_points, n_coords))
        coef = gs.cast(gs.array(n_sampling_points - 1), gs.float32)
        velocity = coef * self.embedding_metric.log(
                point=curve[1:, :], base_point=curve[:-1, :])
        velocity_norm = self.embedding_metric.norm(velocity, curve[:-1, :])
        srv = velocity / gs.sqrt(velocity_norm)

        index = gs.arange(n_curves * n_sampling_points - 1)
        mask = ~gs.equal((index + 1) % n_sampling_points, 0)
        index_select = gs.gather(index, gs.squeeze(gs.where(mask)))
        srv = gs.reshape(gs.gather(srv, index_select), srv_shape)

        return srv

    def square_root_velocity_inverse(self, srv, starting_point):
        """
        Retreive a curve from its square root velocity representation
        and starting point.
        """
        if not isinstance(self.embedding_metric, EuclideanMetric):
            raise AssertionError('The square root velocity inverse is only '
                                 'implemented for dicretized curves embedded '
                                 'in a Euclidean space.')
        if gs.ndim(srv) != gs.ndim(starting_point):
            starting_point = gs.transpose(
                    np.tile(starting_point, (1, 1, 1)),
                    axes=(1, 0, 2))
        srv_shape = srv.shape
        srv = gs.to_ndarray(srv, to_ndim=3)
        n_curves, n_sampling_points_minus_one, n_coords = srv.shape

        srv = gs.reshape(
            srv, (n_curves * n_sampling_points_minus_one, n_coords))
        srv_norm = self.embedding_metric.norm(srv)
        delta_points = 1 / n_sampling_points_minus_one * srv_norm * srv
        delta_points = gs.reshape(delta_points, srv_shape)
        curve = np.concatenate((starting_point, delta_points), -2)
        curve = np.cumsum(curve, -2)

        return curve

    def exp(self, tangent_vec, base_curve):
        """
        Riemannian exponential of a tangent vector wrt to a base curve.
        """
        if not isinstance(self.embedding_metric, EuclideanMetric):
            raise AssertionError('The exponential map is only implemented '
                                 'for dicretized curves embedded in a '
                                 'Euclidean space.')
        base_curve = gs.to_ndarray(base_curve, to_ndim=3)
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
        n_sampling_points = base_curve.shape[1]

        base_curve_srv = self.square_root_velocity(base_curve)

        tangent_vec_derivative = (n_sampling_points - 1) * (
                tangent_vec[:, 1:, :] - tangent_vec[:, :-1, :])
        base_curve_velocity = (n_sampling_points - 1) * (
                base_curve[:, 1:, :] - base_curve[:, :-1, :])
        base_curve_velocity_norm = self.pointwise_norm(
                base_curve_velocity, base_curve[:, :-1, :])

        inner_prod = self.pointwise_inner_product(tangent_vec_derivative,
                                                  base_curve_velocity,
                                                  base_curve[:, :-1, :])
        coef_1 = 1 / gs.sqrt(base_curve_velocity_norm)
        coef_2 = - 1 / (2 * base_curve_velocity_norm ** (5/2)) * inner_prod

        term_1 = gs.einsum('ij,ijk->ijk', coef_1, tangent_vec_derivative)
        term_2 = gs.einsum('ij,ijk->ijk', coef_2, base_curve_velocity)
        srv_initial_derivative = term_1 + term_2

        end_curve_srv = self.l2_metric.exp(tangent_vec=srv_initial_derivative,
                                           base_curve=base_curve_srv)
        end_curve_starting_point = self.embedding_metric.exp(
                tangent_vec=tangent_vec[:, 0, :],
                base_point=base_curve[:, 0, :])
        end_curve = self.square_root_velocity_inverse(
                end_curve_srv, end_curve_starting_point)

        return end_curve

    def log(self, curve, base_curve):
        """
        Riemannian logarithm of a curve wrt a base curve.
        """
        if not isinstance(self.embedding_metric, EuclideanMetric):
            raise AssertionError('The logarithm map is only implemented '
                                 'for dicretized curves embedded in a '
                                 'Euclidean space.')
        curve = gs.to_ndarray(curve, to_ndim=3)
        base_curve = gs.to_ndarray(base_curve, to_ndim=3)
        n_curves, n_sampling_points, n_coords = curve.shape

        curve_srv = self.square_root_velocity(curve)
        base_curve_srv = self.square_root_velocity(base_curve)

        base_curve_velocity = (n_sampling_points - 1) * (
                base_curve[:, 1:, :] - base_curve[:, :-1, :])
        base_curve_velocity_norm = self.pointwise_norm(
                base_curve_velocity, base_curve[:, :-1, :])

        inner_prod = self.pointwise_inner_product(curve_srv - base_curve_srv,
                                                  base_curve_velocity,
                                                  base_curve[:, :-1, :])
        coef_1 = gs.sqrt(base_curve_velocity_norm)
        coef_2 = 1 / base_curve_velocity_norm ** (3/2) * inner_prod

        term_1 = gs.einsum('ij,ijk->ijk', coef_1, curve_srv - base_curve_srv)
        term_2 = gs.einsum('ij,ijk->ijk', coef_2, base_curve_velocity)
        log_derivative = term_1 + term_2

        log_starting_points = self.embedding_metric.log(
                point=curve[:, 0, :], base_point=base_curve[:, 0, :])
        log_starting_points = gs.transpose(
                np.tile(log_starting_points, (1, 1, 1)), (1, 0, 2))

        log_cumsum = gs.hstack([gs.zeros((n_curves, 1, n_coords)),
                                np.cumsum(log_derivative, -2)])
        log = log_starting_points + 1 / (n_sampling_points - 1) * log_cumsum

        return log

    def geodesic(self, initial_curve,
                 end_curve=None, initial_tangent_vec=None):
        """
        Geodesic specified either by an initial curve and an end curve,
        either by an initial curve and an initial tangent vector.
        """
        if not isinstance(self.embedding_metric, EuclideanMetric):
            raise AssertionError('The geodesics are only implemented for '
                                 'dicretized curves embedded in a '
                                 'Euclidean space.')
        curve_ndim = 2
        curve_shape = initial_curve.shape
        initial_curve = gs.to_ndarray(initial_curve,
                                      to_ndim=curve_ndim+1)

        if end_curve is None and initial_tangent_vec is None:
            raise ValueError('Specify an end curve or an initial tangent '
                             'vector to define the geodesic.')
        if end_curve is not None:
            end_curve = gs.to_ndarray(end_curve, to_ndim=curve_ndim+1)
            shooting_tangent_vec = self.log(curve=end_curve,
                                            base_curve=initial_curve)
            if initial_tangent_vec is not None:
                assert gs.allclose(shooting_tangent_vec, initial_tangent_vec)
            initial_tangent_vec = shooting_tangent_vec
        initial_tangent_vec = gs.array(initial_tangent_vec)
        initial_tangent_vec = gs.to_ndarray(initial_tangent_vec,
                                            to_ndim=curve_ndim+1)

        def curve_on_geodesic(t):
            t = gs.cast(t, gs.float32)
            t = gs.to_ndarray(t, to_ndim=1)
            t = gs.to_ndarray(t, to_ndim=2, axis=1)
            new_initial_curve = gs.to_ndarray(
                                          initial_curve,
                                          to_ndim=curve_ndim+1)
            new_initial_tangent_vec = gs.to_ndarray(
                                          initial_tangent_vec,
                                          to_ndim=curve_ndim+1)

            tangent_vecs = gs.einsum('il,nkm->ikm', t, new_initial_tangent_vec)

            curve_shape_at_time_t = gs.hstack([len(t), curve_shape])
            curve_at_time_t = gs.zeros(curve_shape_at_time_t)
            for k in range(len(t)):
                curve_at_time_t[k, :] = self.exp(
                        tangent_vec=tangent_vecs[k, :],
                        base_curve=new_initial_curve)
            return curve_at_time_t

        return curve_on_geodesic

    def dist(self, curve_a, curve_b):
        """
        Geodesic distance between two curves.
        """
        if not isinstance(self.embedding_metric, EuclideanMetric):
            raise AssertionError('The distance is only implemented for '
                                 'dicretized curves embedded in a '
                                 'Euclidean space.')
        assert curve_a.shape == curve_b.shape

        srv_a = self.square_root_velocity(curve_a)
        srv_b = self.square_root_velocity(curve_b)
        dist_starting_points = self.embedding_metric.dist(
                curve_a[0, :], curve_b[0, :])
        dist_srvs = self.l2_metric.dist(srv_a, srv_b)
        dist = gs.sqrt(dist_starting_points ** 2 + dist_srvs ** 2)

        return dist
