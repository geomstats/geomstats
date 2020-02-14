"""Parameterized manifold."""

import math


import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.landmarks import L2Metric
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric

R2 = Euclidean(dimension=2)
R3 = Euclidean(dimension=3)


class DiscretizedCurves(Manifold):
    """Space of discretized curves sampled at points in ambient_manifold."""

    def __init__(self, ambient_manifold):
        """Initialize DiscretizedCurves object."""
        super(DiscretizedCurves, self).__init__(dimension=math.inf)
        self.ambient_manifold = ambient_manifold
        self.l2_metric = L2Metric(self.ambient_manifold)
        self.square_root_velocity_metric = SRVMetric(self.ambient_manifold)

    def belongs(self, point):
        """Test whether a point belongs to the manifold.

        Parameters
        ----------
        point :

        Returns
        -------
        belongs : bool
        """
        belongs = gs.all(self.ambient_manifold.belongs(point))
        belongs = gs.to_ndarray(belongs, to_ndim=1)
        belongs = gs.to_ndarray(belongs, to_ndim=2, axis=1)
        return belongs


class SRVMetric(RiemannianMetric):
    """Elastic metric defined using the Square Root Velocity Function.

    See [Sea2011]_ for details.

    References
    ----------
    .. [Sea2011] Srivastava et al. 2011.
    """

    def __init__(self, ambient_manifold):
        super(SRVMetric, self).__init__(dimension=math.inf,
                                        signature=(math.inf, 0, 0))
        self.ambient_metric = ambient_manifold.metric
        self.l2_metric = L2Metric(ambient_manifold=ambient_manifold)

    def pointwise_inner_product(self, tangent_vec_a, tangent_vec_b,
                                base_curve):
        """Compute the pointwise inner product of pair of tangent vectors.

        Compute the inner product of the components of a (series of)
        pair(s) of tangent vectors at (a) base curve(s).

        Parameters
        ----------
        tangent_vec_a :
        tangent_vec_b :
        base_curve :

        Returns
        -------
        inner_prod :
        """
        base_curve = gs.to_ndarray(base_curve, to_ndim=3)
        tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=3)
        tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=3)

        n_tangent_vecs = tangent_vec_a.shape[0]
        n_sampling_points = tangent_vec_a.shape[1]
        inner_prod = gs.zeros([n_tangent_vecs, n_sampling_points])

        def inner_prod_aux(vec_a, vec_b, curve):
            inner_prod = self.ambient_metric.inner_product(vec_a, vec_b, curve)
            return gs.squeeze(inner_prod)

        inner_prod = gs.vectorize((tangent_vec_a, tangent_vec_b, base_curve),
                                  lambda x, y, z: inner_prod_aux(x, y, z),
                                  dtype=gs.float32,
                                  multiple_args=True,
                                  signature='(i,j),(i,j),(i,j)->(i)')

        return inner_prod

    def pointwise_norm(self, tangent_vec, base_curve):
        """Compute the norm of tangent vector components at base curve.

        TODO: (revise this to refer to action on single elements)
        Compute the norms of the components of a (series of) tangent
        vector(s) at (a) base curve(s).

        Parameters
        ----------
        tangent_vec :
        base_curve :

        Returns
        -------
        norm :
        """
        sq_norm = self.pointwise_inner_product(tangent_vec_a=tangent_vec,
                                               tangent_vec_b=tangent_vec,
                                               base_curve=base_curve)
        return gs.sqrt(sq_norm)

    def square_root_velocity(self, curve):
        """Compute the square root velocity representation of a curve.

        The velocity is computed using the log map. The case of several curves
        is handled through vectorization. In that case, an index selection
        procedure allows to get rid of the log between the end point of
        curve[k, :, :] and the starting point of curve[k + 1, :, :].

        Parameters
        ----------
        curve :

        Returns
        -------
        srv :
        """
        curve = gs.to_ndarray(curve, to_ndim=3)
        n_curves, n_sampling_points, n_coords = curve.shape
        srv_shape = (n_curves, n_sampling_points - 1, n_coords)

        curve = gs.reshape(curve, (n_curves * n_sampling_points, n_coords))
        coef = gs.cast(gs.array(n_sampling_points - 1), gs.float32)
        velocity = coef * self.ambient_metric.log(point=curve[1:, :],
                                                  base_point=curve[:-1, :])
        velocity_norm = self.ambient_metric.norm(velocity, curve[:-1, :])
        srv = velocity / gs.sqrt(velocity_norm)

        index = gs.arange(n_curves * n_sampling_points - 1)
        mask = ~gs.equal((index + 1) % n_sampling_points, 0)
        index_select = gs.gather(index, gs.squeeze(gs.where(mask)))
        srv = gs.reshape(gs.gather(srv, index_select), srv_shape)

        return srv

    def square_root_velocity_inverse(self, srv, starting_point):
        """Retrieve a curve from sqrt velocity rep and starting point.

        Parameters
        ----------
        srv :
        starting_point :

        Returns
        -------
        curve :
        """
        if not isinstance(self.ambient_metric, EuclideanMetric):
            raise AssertionError('The square root velocity inverse is only '
                                 'implemented for dicretized curves embedded '
                                 'in a Euclidean space.')
        if gs.ndim(srv) != gs.ndim(starting_point):
            starting_point = gs.transpose(
                gs.tile(starting_point, (1, 1, 1)),
                axes=(1, 0, 2))
        srv_shape = srv.shape
        srv = gs.to_ndarray(srv, to_ndim=3)
        n_curves, n_sampling_points_minus_one, n_coords = srv.shape

        srv = gs.reshape(srv,
                         (n_curves * n_sampling_points_minus_one, n_coords))
        srv_norm = self.ambient_metric.norm(srv)
        delta_points = 1 / n_sampling_points_minus_one * srv_norm * srv
        delta_points = gs.reshape(delta_points, srv_shape)
        curve = gs.concatenate((starting_point, delta_points), -2)
        curve = gs.cumsum(curve, -2)

        return curve

    def exp(self, tangent_vec, base_curve):
        """Compute Riemannian exponential of tangent vector wrt to base curve.

        Parameters
        ----------
        tangent_vec :
        base_curve :

        Return
        ------
        end_curve :
        """
        if not isinstance(self.ambient_metric, EuclideanMetric):
            raise AssertionError('The exponential map is only implemented '
                                 'for dicretized curves embedded in a '
                                 'Euclidean space.')
        base_curve = gs.to_ndarray(base_curve, to_ndim=3)
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
        n_sampling_points = base_curve.shape[1]

        base_curve_srv = self.square_root_velocity(base_curve)

        tangent_vec_derivative = (n_sampling_points - 1) * (
            tangent_vec[:, 1:, :] - tangent_vec[:, :-1, :])
        base_curve_velocity = (n_sampling_points - 1) * (base_curve[:, 1:, :] -
                                                         base_curve[:, :-1, :])
        base_curve_velocity_norm = self.pointwise_norm(base_curve_velocity,
                                                       base_curve[:, :-1, :])

        inner_prod = self.pointwise_inner_product(tangent_vec_derivative,
                                                  base_curve_velocity,
                                                  base_curve[:, :-1, :])
        coef_1 = 1 / gs.sqrt(base_curve_velocity_norm)
        coef_2 = -1 / (2 * base_curve_velocity_norm**(5 / 2)) * inner_prod

        term_1 = gs.einsum('ij,ijk->ijk', coef_1, tangent_vec_derivative)
        term_2 = gs.einsum('ij,ijk->ijk', coef_2, base_curve_velocity)
        srv_initial_derivative = term_1 + term_2

        end_curve_srv = self.l2_metric.exp(tangent_vec=srv_initial_derivative,
                                           base_landmarks=base_curve_srv)
        end_curve_starting_point = self.ambient_metric.exp(
            tangent_vec=tangent_vec[:, 0, :], base_point=base_curve[:, 0, :])
        end_curve = self.square_root_velocity_inverse(
            end_curve_srv, end_curve_starting_point)

        return end_curve

    def log(self, curve, base_curve):
        """Compute Riemannian logarithm of a curve wrt a base curve.

        Parameters
        ----------
        curve :
        base_curve :

        Returns
        -------
        log :
        """
        if not isinstance(self.ambient_metric, EuclideanMetric):
            raise AssertionError('The logarithm map is only implemented '
                                 'for dicretized curves embedded in a '
                                 'Euclidean space.')
        curve = gs.to_ndarray(curve, to_ndim=3)
        base_curve = gs.to_ndarray(base_curve, to_ndim=3)
        n_curves, n_sampling_points, n_coords = curve.shape

        curve_srv = self.square_root_velocity(curve)
        base_curve_srv = self.square_root_velocity(base_curve)

        base_curve_velocity = (n_sampling_points - 1) * (base_curve[:, 1:, :] -
                                                         base_curve[:, :-1, :])
        base_curve_velocity_norm = self.pointwise_norm(base_curve_velocity,
                                                       base_curve[:, :-1, :])

        inner_prod = self.pointwise_inner_product(curve_srv - base_curve_srv,
                                                  base_curve_velocity,
                                                  base_curve[:, :-1, :])
        coef_1 = gs.sqrt(base_curve_velocity_norm)
        coef_2 = 1 / base_curve_velocity_norm**(3 / 2) * inner_prod

        term_1 = gs.einsum('ij,ijk->ijk', coef_1, curve_srv - base_curve_srv)
        term_2 = gs.einsum('ij,ijk->ijk', coef_2, base_curve_velocity)
        log_derivative = term_1 + term_2

        log_starting_points = self.ambient_metric.log(
            point=curve[:, 0, :], base_point=base_curve[:, 0, :])
        log_starting_points = gs.transpose(
            gs.tile(log_starting_points, (1, 1, 1)), (1, 0, 2))

        log_cumsum = gs.hstack(
            [gs.zeros((n_curves, 1, n_coords)),
             gs.cumsum(log_derivative, -2)])
        log = log_starting_points + 1 / (n_sampling_points - 1) * log_cumsum

        return log

    def geodesic(self,
                 initial_curve,
                 end_curve=None,
                 initial_tangent_vec=None):
        """Compute geodesic from initial curve and end curve end curve.

        Geodesic specified either by an initial curve and an end curve,
        either by an initial curve and an initial tangent vector.

        Parameters
        ----------
        initial_curve :
        end_curve :
        inital_tangent_vec :

        Returns
        -------
        curve_on_geodesic :
        """
        if not isinstance(self.ambient_metric, EuclideanMetric):
            raise AssertionError('The geodesics are only implemented for '
                                 'dicretized curves embedded in a '
                                 'Euclidean space.')
        curve_ndim = 2
        curve_shape = initial_curve.shape
        initial_curve = gs.to_ndarray(initial_curve, to_ndim=curve_ndim + 1)

        if end_curve is None and initial_tangent_vec is None:
            raise ValueError('Specify an end curve or an initial tangent '
                             'vector to define the geodesic.')
        if end_curve is not None:
            end_curve = gs.to_ndarray(end_curve, to_ndim=curve_ndim + 1)
            shooting_tangent_vec = self.log(curve=end_curve,
                                            base_curve=initial_curve)
            if initial_tangent_vec is not None:
                assert gs.allclose(shooting_tangent_vec, initial_tangent_vec)
            initial_tangent_vec = shooting_tangent_vec
        initial_tangent_vec = gs.array(initial_tangent_vec)
        initial_tangent_vec = gs.to_ndarray(initial_tangent_vec,
                                            to_ndim=curve_ndim + 1)

        def curve_on_geodesic(t):
            t = gs.cast(t, gs.float32)
            t = gs.to_ndarray(t, to_ndim=1)
            t = gs.to_ndarray(t, to_ndim=2, axis=1)
            new_initial_curve = gs.to_ndarray(initial_curve,
                                              to_ndim=curve_ndim + 1)
            new_initial_tangent_vec = gs.to_ndarray(initial_tangent_vec,
                                                    to_ndim=curve_ndim + 1)

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
        """Geodesic distance between two curves.

        Parameters
        ----------
        curve_a :
        curve_b :

        Returns
        -------
        dist :
        """
        if not isinstance(self.ambient_metric, EuclideanMetric):
            raise AssertionError('The distance is only implemented for '
                                 'dicretized curves embedded in a '
                                 'Euclidean space.')
        assert curve_a.shape == curve_b.shape

        srv_a = self.square_root_velocity(curve_a)
        srv_b = self.square_root_velocity(curve_b)
        dist_starting_points = self.ambient_metric.dist(
            curve_a[0, :], curve_b[0, :])
        dist_srvs = self.l2_metric.dist(srv_a, srv_b)
        dist = gs.sqrt(dist_starting_points**2 + dist_srvs**2)

        return dist
