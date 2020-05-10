"""Parameterized curves on any given manifold."""

import math


import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.landmarks import L2Metric
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric

R2 = Euclidean(dim=2)
R3 = Euclidean(dim=3)


class DiscreteCurves(Manifold):
    r"""Space of discrete curves sampled at points in ambient_manifold.

    Each individual curve is represented by a 2d-array of shape `[
    n_sampling_points, ambient_dim]`. A Batch of curves can be passed to
    all methods either as a 3d-array if all curves have the same number of
    sampled points, or as a list of 2d-arrays, each representing a curve.

    Parameters
    ----------
    ambient_manifold : Manifold
        Manifold in which curves take values.

    Attributes
    ----------
    ambient_manifold : Manifold
        Manifold in which curves take values.
    l2_metric : callable
        Function that takes as argument an integer number of sampled points
        and returns the corresponding L2 metric (product) metric,
        a RiemannianMetric object
    square_root_velocity_metric : RiemannianMetric
        Square root velocity metric.
    """

    def __init__(self, ambient_manifold):
        super(DiscreteCurves, self).__init__(dim=math.inf)
        self.ambient_manifold = ambient_manifold
        self.l2_metric = lambda n: L2Metric(
            self.ambient_manifold, n_landmarks=n)
        self.square_root_velocity_metric = SRVMetric(self.ambient_manifold)

    def belongs(self, point):
        """Test whether a point belongs to the manifold.

        Test that all points of the curve belong to the ambient manifold.

        Parameters
        ----------
        point : array-like, shape=[..., n_sampling_points, ambient_dim]
            Point representing a discrete curve.

        Returns
        -------
        belongs : bool
            Boolean evaluating if point belongs to the space of discrete
            curves.
        """
        def each_belongs(pt):
            return gs.all(self.ambient_manifold.belongs(pt))

        if isinstance(point, list) or point.ndim > 2:
            return gs.stack([each_belongs(pt) for pt in point])

        return each_belongs(point)


class SRVMetric(RiemannianMetric):
    """Elastic metric defined using the Square Root Velocity Function.

    See [Sea2011]_ for details.

    Parameters
    ----------
    ambient_manifold : Manifold
        Manifold in which curves take values.
    metric : RiemannianMetric
        Metric to use on the ambient manifold. If None is passed, ambient
        manifold should have a metric attribute, which will be used.
        Optional, default : None.

    References
    ----------
    .. [Sea2011] A. Srivastava, E. Klassen, S. H. Joshi and I. H. Jermyn,
    "Shape Analysis of Elastic Curves in Euclidean Spaces,"
    in IEEE Transactions on Pattern Analysis and Machine Intelligence,
    vol. 33, no. 7, pp. 1415-1428, July 2011.
    """

    def __init__(self, ambient_manifold, metric=None):
        super(SRVMetric, self).__init__(dim=math.inf,
                                        signature=(math.inf, 0, 0))
        if metric is None:
            if hasattr(ambient_manifold, 'metric'):
                self.ambient_metric = ambient_manifold.metric
            else:
                raise ValueError('Instantiating an object of class '
                                 'DiscreteCurves requires either a metric'
                                 ' or an ambient manifold'
                                 ' equipped with a metric.')
        else:
            self.ambient_metric = metric
        self.l2_metric = lambda n: L2Metric(ambient_manifold, n_landmarks=n)

    def pointwise_inner_product(self, tangent_vec_a, tangent_vec_b,
                                base_curve):
        """Compute the pointwise inner product of pair of tangent vectors.

        Compute the point-wise inner-product between two tangent vectors
        at a base curve.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n_sampling_points, ambient_dim]
            Tangent vector to discrete curve.
        tangent_vec_b : array-like, shape=[..., n_sampling_points, ambient_dim]
            Tangent vector to discrete curve.
        base_curve : array-like, shape=[..., n_sampling_points, ambient_dim]
            Point representing a discrete curve.

        Returns
        -------
        inner_prod : array-like, shape=[..., n_sampling_points]
            Point-wise inner-product.
        """
        def inner_prod_aux(vec_a, vec_b, curve):
            inner_prod = self.ambient_metric.inner_product(vec_a, vec_b, curve)
            return gs.squeeze(inner_prod)

        inner_prod = gs.vectorize(
            (tangent_vec_a, tangent_vec_b, base_curve),
            inner_prod_aux,
            dtype=gs.float32,
            multiple_args=True,
            signature='(i,j),(i,j),(i,j)->(i)')

        return inner_prod

    def pointwise_norm(self, tangent_vec, base_curve):
        """Compute the point-wise norm of a tangent vector at a base curve.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n_sampling_points, ambient_dim]
            Tangent vector to discrete curve.
        base_curve : array-like, shape=[..., n_sampling_points, ambient_dim]
            Point representing a discrete curve.

        Returns
        -------
        norm : array-like, shape=[..., n_sampling_points]
            Point-wise norms.
        """
        sq_norm = self.pointwise_inner_product(
            tangent_vec_a=tangent_vec, tangent_vec_b=tangent_vec,
            base_curve=base_curve)
        return gs.sqrt(sq_norm)

    def square_root_velocity(self, curve):
        """Compute the square root velocity representation of a curve.

        The velocity is computed using the log map. In the case of several
        curves, an index selection procedure allows to get rid of the log
        between the end point of curve[k, :, :] and the starting point of
        curve[k + 1, :, :].

        Parameters
        ----------
        curve : array-like, shape=[..., n_sampling_points, ambient_dim]
            Discrete curve.

        Returns
        -------
        srv : array-like, shape=[..., n_sampling_points - 1, ambient_dim]
            Square-root velocity representation of a discrete curve.
        """
        curve = gs.to_ndarray(curve, to_ndim=3)
        n_curves, n_sampling_points, n_coords = curve.shape
        srv_shape = (n_curves, n_sampling_points - 1, n_coords)

        curve = gs.reshape(curve, (n_curves * n_sampling_points, n_coords))
        coef = gs.cast(gs.array(n_sampling_points - 1), gs.float32)
        velocity = coef * self.ambient_metric.log(point=curve[1:, :],
                                                  base_point=curve[:-1, :])
        velocity_norm = self.ambient_metric.norm(velocity, curve[:-1, :])
        srv = gs.einsum(
            '...i,...->...i', velocity, 1. / gs.sqrt(velocity_norm))

        index = gs.arange(n_curves * n_sampling_points - 1)
        mask = ~((index + 1) % n_sampling_points == 0)
        srv = gs.reshape(srv[mask], srv_shape)

        return srv

    def square_root_velocity_inverse(self, srv, starting_point):
        """Retrieve a curve from sqrt velocity rep and starting point.

        Parameters
        ----------
        srv : array-like, shape=[..., n_sampling_points - 1, ambient_dim]
            Square-root velocity representation of a discrete curve.
        starting_point : array-like, shape=[..., ambient_dim]
            Point of the ambient manifold to use as start of the retrieved
            curve.

        Returns
        -------
        curve : array-like, shape=[..., n_sampling_points, ambient_dim]
            Curve retrieved from its square-root velocity.
        """
        if not isinstance(self.ambient_metric, EuclideanMetric):
            raise AssertionError('The square root velocity inverse is only '
                                 'implemented for discrete curves embedded '
                                 'in a Euclidean space.')
        if gs.ndim(srv) != gs.ndim(starting_point):
            starting_point = gs.to_ndarray(
                starting_point, to_ndim=srv.ndim, axis=1)
        srv_shape = srv.shape
        srv = gs.to_ndarray(srv, to_ndim=3)
        n_curves, n_sampling_points_minus_one, n_coords = srv.shape

        srv = gs.reshape(srv,
                         (n_curves * n_sampling_points_minus_one, n_coords))
        srv_norm = self.ambient_metric.norm(srv)
        delta_points = gs.einsum(
            '...,...i->...i', 1 / n_sampling_points_minus_one * srv_norm, srv)
        delta_points = gs.reshape(delta_points, srv_shape)
        curve = gs.concatenate((starting_point, delta_points), -2)
        curve = gs.cumsum(curve, -2)

        return curve

    def exp(self, tangent_vec, base_point):
        """Compute Riemannian exponential of tangent vector wrt to base curve.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n_sampling_points, ambient_dim]
            Tangent vector to discrete curve.
        base_point : array-like, shape=[..., n_sampling_points, ambient_dim]
            Discrete curve.

        Return
        ------
        end_curve :  array-like, shape=[..., n_sampling_points, ambient_dim]
            Discrete curve, result of the Riemannian exponential.
        """
        if not isinstance(self.ambient_metric, EuclideanMetric):
            raise AssertionError('The exponential map is only implemented '
                                 'for discrete curves embedded in a '
                                 'Euclidean space.')
        base_point = gs.to_ndarray(base_point, to_ndim=3)
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
        n_sampling_points = base_point.shape[1]

        base_curve_srv = self.square_root_velocity(base_point)

        tangent_vec_derivative = (n_sampling_points - 1) * (
            tangent_vec[:, 1:, :] - tangent_vec[:, :-1, :])
        base_curve_velocity = (n_sampling_points - 1) * (
            base_point[:, 1:, :] - base_point[:, :-1, :])
        base_curve_velocity_norm = self.pointwise_norm(
            base_curve_velocity, base_point[:, :-1, :])

        inner_prod = self.pointwise_inner_product(
            tangent_vec_derivative, base_curve_velocity, base_point[:, :-1, :])
        coef_1 = 1 / gs.sqrt(base_curve_velocity_norm)
        coef_2 = -1 / (2 * base_curve_velocity_norm**(5 / 2)) * inner_prod

        term_1 = gs.einsum('ij,ijk->ijk', coef_1, tangent_vec_derivative)
        term_2 = gs.einsum('ij,ijk->ijk', coef_2, base_curve_velocity)
        srv_initial_derivative = term_1 + term_2

        end_curve_srv = self.l2_metric(n_sampling_points - 1).exp(
            tangent_vec=srv_initial_derivative, base_point=base_curve_srv)
        end_curve_starting_point = self.ambient_metric.exp(
            tangent_vec=tangent_vec[:, 0, :], base_point=base_point[:, 0, :])
        end_curve = self.square_root_velocity_inverse(
            end_curve_srv, end_curve_starting_point)

        return end_curve

    def log(self, point, base_point):
        """Compute Riemannian logarithm of a curve wrt a base curve.

        Parameters
        ----------
        point : array-like, shape=[..., n_sampling_points, ambient_dim]
            Discrete curve.
        base_point : array-like, shape=[..., n_sampling_points, ambient_dim]
            Discrete curve to use as base point.

        Returns
        -------
        log : array-like, shape=[..., n_sampling_points, ambient_dim]
            Tangent vector to a discrete curve.
        """
        if not isinstance(self.ambient_metric, EuclideanMetric):
            raise AssertionError('The logarithm map is only implemented '
                                 'for discrete curves embedded in a '
                                 'Euclidean space.')
        point = gs.to_ndarray(point, to_ndim=3)
        base_point = gs.to_ndarray(base_point, to_ndim=3)
        n_curves, n_sampling_points, n_coords = point.shape

        curve_srv = self.square_root_velocity(point)
        base_curve_srv = self.square_root_velocity(base_point)

        base_curve_velocity = (n_sampling_points - 1) * (base_point[:, 1:, :] -
                                                         base_point[:, :-1, :])
        base_curve_velocity_norm = self.pointwise_norm(base_curve_velocity,
                                                       base_point[:, :-1, :])

        inner_prod = self.pointwise_inner_product(curve_srv - base_curve_srv,
                                                  base_curve_velocity,
                                                  base_point[:, :-1, :])
        coef_1 = gs.sqrt(base_curve_velocity_norm)
        coef_2 = 1 / base_curve_velocity_norm**(3 / 2) * inner_prod

        term_1 = gs.einsum('ij,ijk->ijk', coef_1, curve_srv - base_curve_srv)
        term_2 = gs.einsum('ij,ijk->ijk', coef_2, base_curve_velocity)
        log_derivative = term_1 + term_2

        log_starting_points = self.ambient_metric.log(
            point=point[:, 0, :], base_point=base_point[:, 0, :])
        log_starting_points = gs.to_ndarray(
            log_starting_points, to_ndim=3, axis=1)

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
        initial_curve : array-like, shape=[..., n_sampling_points, ambient_dim]
            Discrete curve.
        end_curve : array-like, shape=[..., n_sampling_points, ambient_dim]
            Discrete curve. If None, an initial tangent vector must be given.
            Optional, default : None
        initial_tangent_vec : array-like,
            shape=[..., n_sampling_points, ambient_dim]
            Tangent vector at base curve, the initial speed of the geodesics.
            If None, an end curve must be given and a logarithm is computed.
            Optional, default : None

        Returns
        -------
        curve_on_geodesic : callable
            The time parameterized geodesic curve.
        """
        if not isinstance(self.ambient_metric, EuclideanMetric):
            raise AssertionError('The geodesics are only implemented for '
                                 'discrete curves embedded in a '
                                 'Euclidean space.')
        curve_ndim = 2
        initial_curve = gs.to_ndarray(initial_curve, to_ndim=curve_ndim + 1)

        if end_curve is None and initial_tangent_vec is None:
            raise ValueError('Specify an end curve or an initial tangent '
                             'vector to define the geodesic.')
        if end_curve is not None:
            end_curve = gs.to_ndarray(end_curve, to_ndim=curve_ndim + 1)
            shooting_tangent_vec = self.log(point=end_curve,
                                            base_point=initial_curve)
            if initial_tangent_vec is not None:
                if not gs.allclose(shooting_tangent_vec, initial_tangent_vec):
                    raise RuntimeError(
                        'The shooting tangent vector is too'
                        ' far from the initial tangent vector.')
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

            curve_at_time_t = []
            for k in range(len(t)):
                curve_at_time_t.append(self.exp(
                    tangent_vec=tangent_vecs[k, :],
                    base_point=new_initial_curve))
            return gs.stack(curve_at_time_t)

        return curve_on_geodesic

    def dist(self, point_a, point_b):
        """Geodesic distance between two curves.

        Parameters
        ----------
        point_a : array-like, shape=[..., n_sampling_points, ambient_dim]
            Discrete curve.
        point_b : array-like, shape=[..., n_sampling_points, ambient_dim]
            Discrete curve.

        Returns
        -------
        dist : array-like, shape=[...,]
        """
        if not isinstance(self.ambient_metric, EuclideanMetric):
            raise AssertionError('The distance is only implemented for '
                                 'discrete curves embedded in a '
                                 'Euclidean space.')
        if point_a.shape != point_b.shape:
            raise ValueError('The curves need to have the same shapes.')

        srv_a = self.square_root_velocity(point_a)
        srv_b = self.square_root_velocity(point_b)
        n_sampling_points = srv_a.shape[-2]
        dist_starting_points = self.ambient_metric.dist(
            point_a[0, :], point_b[0, :])
        dist_srvs = self.l2_metric(n_sampling_points).dist(srv_a, srv_b)
        dist = gs.sqrt(dist_starting_points**2 + dist_srvs**2)

        return dist
