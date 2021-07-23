"""Parameterized curves on any given manifold."""

import math


import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.landmarks import L2Metric
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.symmetric_matrices import SymmetricMatrices

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

    def belongs(self, point, atol=gs.atol):
        """Test whether a point belongs to the manifold.

        Test that all points of the curve belong to the ambient manifold.

        Parameters
        ----------
        point : array-like, shape=[..., n_sampling_points, ambient_dim]
            Point representing a discrete curve.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

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

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at a curve.

        A vector is tangent at a curve if it is a vector field along that
        curve.

        Parameters
        ----------
        vector : array-like, shape=[..., n_sampling_points, ambient_dim]
            Vector.
        base_point : array-like, shape=[..., n_sampling_points, ambient_dim]
            Discrete curve.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        ambient_manifold = self.ambient_manifold
        shape = vector.shape
        stacked_vec = gs.reshape(vector, (-1, shape[-1]))
        stacked_point = gs.reshape(base_point, (-1, shape[-1]))
        is_tangent = ambient_manifold.is_tangent(
            stacked_vec, stacked_point, atol)
        is_tangent = gs.reshape(is_tangent, shape[:-1])
        return gs.all(is_tangent, axis=-1)

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        As tangent vectors are vector fields along a curve, each component of
        the vector is projected to the tangent space of the corresponding
        point of the discrete curve. The number of sampling points should
        match in the vector and the base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., n_sampling_points, ambient_dim]
            Vector.
        base_point : array-like, shape=[..., n_sampling_points, ambient_dim]
            Discrete curve.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n_sampling_points, ambient_dim]
            Tangent vector at base point.
        """
        ambient_manifold = self.ambient_manifold
        shape = vector.shape
        stacked_vec = gs.reshape(vector, (-1, shape[-1]))
        stacked_point = gs.reshape(base_point, (-1, shape[-1]))
        tangent_vec = ambient_manifold.to_tangent(stacked_vec, stacked_point)
        tangent_vec = gs.reshape(tangent_vec, vector.shape)
        return tangent_vec

    def random_point(self, n_samples=1, bound=1., n_sampling_points=10):
        """Sample random curves.

        If the ambient manifold is compact, a uniform distribution is used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample for non compact
            ambient manifolds.
            Optional, default: 1.
        n_sampling_points : int
            Number of sampling points for the discrete curves.
            Optional, default : 10.

        Returns
        -------
        samples : array-like, shape=[..., n_sampling_points, {dim, [n, n]}]
            Points sampled on the hypersphere.
        """
        sample = self.ambient_manifold.random_point(
            n_samples * n_sampling_points)
        sample = gs.reshape(sample, (n_samples, n_sampling_points, -1))
        return sample[0] if n_samples == 1 else sample


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

    def srv_inner_product(self, srv_1, srv_2):
        """
        Compute the L² inner_product between two srv representations.

        Parameters
        ----------
        srv_1 : array-like, shape=[..., n_sampling_points, ambient_dim]
            Srv representation.
        srv_2 : array-like, shape=[..., n_sampling_points, ambient_dim]
            Srv representation.

        Return
        ------
        inner_prod : array-like, shape=[...]
            L² inner product between the two srv representations.
        """
        n_sampling_points = srv_1.shape[-2]

        l2_inner_prod = self.l2_metric(n_sampling_points).inner_product
        inner_prod = l2_inner_prod(srv_1, srv_2) / (n_sampling_points)

        return inner_prod

    def srv_norm(self, srv):
        """
        Compute the L² norm of a srv representation of a curve.

        Parameters
        ----------
        srv : array-like, shape=[..., n_sampling_points, ambient_dim]
            Srv representation of a curve

        Return
        ------
        norm : array-like, shape=[...]
            L² norm of the srv representation.
        """
        squared_norm = self.srv_inner_product(srv, srv)
        norm = gs.sqrt(squared_norm)

        return norm

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
            for tan_vec in tangent_vecs:
                curve_at_time_t.append(
                    self.exp(tan_vec, new_initial_curve))
            return gs.stack(curve_at_time_t)

        return curve_on_geodesic

    def dist(self, point_a, point_b, **kwargs):
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


class ClosedDiscreteCurves(Manifold):
    r"""Space of closed discrete curves sampled at points in ambient_manifold.

    Each individual curve is represented by a 2d-array of shape `[
    n_sampling_points, ambient_dim]`.

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
        super(ClosedDiscreteCurves, self).__init__(dim=math.inf)
        self.ambient_manifold = ambient_manifold
        self.l2_metric = lambda n: L2Metric(
            self.ambient_manifold, n_landmarks=n)
        self.square_root_velocity_metric = ClosedSRVMetric(ambient_manifold)

    def belongs(self, point, atol=gs.atol):
        """Test whether a point belongs to the manifold.

        Test that all points of the curve belong to the ambient manifold.

        Parameters
        ----------
        point : array-like, shape=[..., n_sampling_points, ambient_dim]
            Point representing a discrete curve.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : bool
            Boolean evaluating if point belongs to the space of discrete
            curves.
        """
        raise NotImplementedError(
            'The belongs method is not implemented.'
        )

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at a curve.

        A vector is tangent at a curve if it is a vector field along that
        curve.

        Parameters
        ----------
        vector : array-like, shape=[..., n_sampling_points, ambient_dim]
            Vector.
        base_point : array-like, shape=[..., n_sampling_points, ambient_dim]
            Discrete curve.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        raise NotImplementedError(
            'The is_tangent method is not implemented.'
        )

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        As tangent vectors are vector fields along a curve, each component of
        the vector is projected to the tangent space of the corresponding
        point of the discrete curve. The number of sampling points should
        match in the vector and the base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., n_sampling_points, ambient_dim]
            Vector.
        base_point : array-like, shape=[..., n_sampling_points, ambient_dim]
            Discrete curve.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n_sampling_points, ambient_dim]
            Tangent vector at base point.
        """
        raise NotImplementedError(
            'The to_tangent method is not implemented.'
        )

    def random_point(self, n_samples=1, bound=1., n_sampling_points=10):
        """Sample random curves.

        If the ambient manifold is compact, a uniform distribution is used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample for non compact
            ambient manifolds.
            Optional, default: 1.
        n_sampling_points : int
            Number of sampling points for the discrete curves.
            Optional, default : 10.

        Returns
        -------
        samples : array-like, shape=[..., n_sampling_points, {dim, [n, n]}]
            Points sampled on the hypersphere.
        """
        raise NotImplementedError(
            'The random_point method is not implemented.'
        )

    def project(self, curve, atol=gs.atol, max_iter=1000):
        """Project a discrete curve into the space of closed discrete curves.

        Parameters
        ----------
        curve : array-like, shape=[..., n_sampling_points, ambient_dim]
            Discrete curve.
        atol : float
            Tolerance of the projection algorithm.
            Optional, default: backend atol.
        max_iter : float
            Maximum number of iteration of the algorithm.
            Optional, default: 1000

        Returns
        -------
        proj : array-like, shape=[..., n_sampling_points, ambient_dim]
        """
        is_euclidean = isinstance(self.ambient_manifold, Euclidean)
        is_planar = is_euclidean and self.ambient_manifold.dim == 2

        if not is_planar:
            raise AssertionError('The projection is only implemented '
                                 'for discrete curves embedded in a'
                                 '2D Euclidean space.')

        srv_metric = self.square_root_velocity_metric
        srv = srv_metric.square_root_velocity(curve)[0]
        srv_proj = srv_metric.project_srv(srv, atol=atol, max_iter=max_iter)
        proj = srv_metric.square_root_velocity_inverse(srv_proj,
                                                       gs.array([curve[0]]))

        return proj


class ClosedSRVMetric(SRVMetric):
    """Elastic metric on closed curves.

    See [Sea2011]_ for details.

    Parameters
    ----------
    ambient_manifold : Manifold
        Manifold in which curves take values.

    References
    ----------
    .. [Sea2011] A. Srivastava, E. Klassen, S. H. Joshi and I. H. Jermyn,
    "Shape Analysis of Elastic Curves in Euclidean Spaces,"
    in IEEE Transactions on Pattern Analysis and Machine Intelligence,
    vol. 33, no. 7, pp. 1415-1428, July 2011.
    """

    def __init__(self, ambient_manifold):
        super(ClosedSRVMetric, self).__init__(ambient_manifold)

    def project_srv(self, srv, atol=gs.atol, max_iter=1000):
        """Project a point in the srv space into the space of closed curves srv.

        The algorithm is from the paper cited above and modifies the srv
        iteratively so that G(srv) = (0, ..., 0) with the paper's notation.

        Remark: for now, the algorithm might not converge for some curves such
        as segments.

        Parameters
        ----------
        srv : array-like, shape=[..., n_sampling_points, ambient_dim]
            Discrete curve.
        atol : float
            Tolerance of the projection algorithm.
            Optional, default: backend atol.
        max_iter : float
            Maximum number of iteration of the algorithm.
            Optional, default: 1000

        Returns
        -------
        proj : array-like, shape=[..., n_sampling_points, ambient_dim]
        """
        is_euclidean = isinstance(self.ambient_metric, EuclideanMetric)
        is_planar = is_euclidean and self.ambient_metric.dim == 2

        if not is_planar:
            raise AssertionError('The projection is only implemented '
                                 'for discrete curves embedded in a'
                                 '2D Euclidean space.')

        dim = self.ambient_metric.dim
        srv_inner_prod = self.srv_inner_product
        srv_norm = self.srv_norm
        inner_prod = self.ambient_metric.inner_product

        def g_criterion(srv, srv_norms):
            return gs.sum(srv * srv_norms[:, None], axis=0)

        initial_norm = srv_norm(srv)
        proj = srv
        proj_norms = self.ambient_metric.norm(proj)
        residual = g_criterion(proj, proj_norms)
        criteria = self.ambient_metric.norm(residual)

        nb_iter = 0

        while criteria >= atol and nb_iter < max_iter:

            jacobian_vec = []
            for i in range(dim):
                for j in range(i, dim):
                    coef = 3 * inner_prod(proj[:, i], proj[:, j])
                    jacobian_vec.append(coef)
            jacobian_vec = gs.stack(jacobian_vec)
            g_jacobian = SymmetricMatrices.from_vector(jacobian_vec)

            proj_squared_norm = srv_norm(proj) ** 2
            g_jacobian += proj_squared_norm * gs.eye(dim)
            beta = gs.linalg.inv(g_jacobian) @ residual

            e_1, e_2 = gs.array([1, 0]), gs.array([0, 1])
            grad_1 = proj_norms[:, None] * e_1
            grad_1 = grad_1 + (proj[:, 0] / proj_norms)[:, None] * proj
            grad_2 = proj_norms[:, None] * e_2
            grad_2 = grad_2 + (proj[:, 1] / proj_norms)[:, None] * proj

            basis_vector_1 = grad_1 / srv_norm(grad_1)
            grad_2_component = srv_inner_prod(grad_2, basis_vector_1)
            grad_2_proj = grad_2_component * basis_vector_1
            basis_vector_2 = grad_2 - grad_2_proj
            basis_vector_2 = basis_vector_2 / srv_norm(basis_vector_2)
            basis = gs.array([basis_vector_1, basis_vector_2])

            proj -= gs.sum(beta[:, None, None] * basis, axis=0)
            proj = proj * initial_norm / srv_norm(proj)
            proj_norms = self.ambient_metric.norm(proj)
            residual = g_criterion(proj, proj_norms)
            criteria = self.ambient_metric.norm(residual)

            nb_iter += 1

        return proj
