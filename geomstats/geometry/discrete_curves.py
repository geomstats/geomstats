"""Parameterized curves on any given manifold.

Lead author: Alice Le Brigant.
"""

import math

from scipy.interpolate import CubicSpline

import geomstats.backend as gs
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.base import LevelSet
from geomstats.geometry.euclidean import Euclidean, EuclideanMetric
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.landmarks import L2LandmarksMetric
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.symmetric_matrices import SymmetricMatrices

R2 = Euclidean(dim=2)
R3 = Euclidean(dim=3)


class DiscreteCurves(Manifold):
    r"""Space of discrete curves sampled at points in ambient_manifold.

    Each individual curve is represented by a 2d-array of shape `[
    k_sampling_points, ambient_dim]`. A batch of curves can be passed to
    all methods either as a 3d-array if all curves have the same number of
    sampled points, or as a list of 2d-arrays, each representing a curve.

    This space corresponds to the space of immersions defined below, i.e. the
    space of smooth functions from an interval I into the ambient manifold M,
    with non-vanishing derivative.

    .. math::
        Imm(I, M)=\{c \in C^{\infty}(I, M) \|c'(t)\|\neq 0 \forall t \in I \},

    where the open interval of parameters I is taken to be I = [0, 1]
    without loss of generality.

    Parameters
    ----------
    ambient_manifold : Manifold
        Manifold in which curves take values.

    Attributes
    ----------
    ambient_manifold : Manifold
        Manifold in which curves take values.
    l2_landmarks_metric : callable
        Function that takes as argument an integer number of sampled points
        and returns the corresponding L2 metric (product) metric,
        a RiemannianMetric object
    srv_metric : RiemannianMetric
        Square root velocity metric.
    """

    def __init__(
        self, ambient_manifold, k_sampling_points=10, a=None, b=None, **kwargs
    ):
        dim = ambient_manifold.dim * k_sampling_points
        kwargs.setdefault("metric", SRVMetric(ambient_manifold))
        super(DiscreteCurves, self).__init__(
            dim=dim,
            shape=(k_sampling_points,) + ambient_manifold.shape,
            **kwargs,
        )
        self.ambient_manifold = ambient_manifold
        self.k_sampling_points = k_sampling_points
        self.l2_curves_metric = L2CurvesMetric(ambient_manifold=ambient_manifold)
        self.srv_metric = self._metric

        if a is not None and b is not None:
            self.elastic_metric = ElasticMetric(
                a=a, b=b, ambient_manifold=ambient_manifold
            )

    def belongs(self, point, atol=gs.atol):
        """Test whether a point belongs to the manifold.

        Test that all points of the curve belong to the ambient manifold.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points, ambient_dim]
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
            """Check that sampling points are in ambient manifold.

            Parameters
            ----------
            pt : array-like, shape=[k_sampling_points, ambient_dim]
                One curve represented as its sampling points.

            Returns
            -------
            _ : array-like, shape=[]
                Whether curve has all of its sampling points on
                the ambient manifold.
            """
            return gs.all(self.ambient_manifold.belongs(pt))

        def each_has_k_sampling_points(pt):
            """Check that a curve has the correct number of sampling points.

            Parameters
            ----------
            pt : array-like, shape=[k_sampling_points_to_test, ambient_dim]
                One curve represented as its sampling points.

            Returns
            -------
            _ : array-like, shape=[]
                Whether curve has the correct number of sampling points.
            """
            return gs.array(pt.shape[-2] == self.k_sampling_points)

        if isinstance(point, list) or point.ndim > 2:
            return gs.stack(
                [each_belongs(pt) and each_has_k_sampling_points(pt) for pt in point]
            )

        return each_belongs(point) and each_has_k_sampling_points(point)

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at a curve.

        A vector is tangent at a curve if it is a vector field along that
        curve.

        Parameters
        ----------
        vector : array-like, shape=[..., k_sampling_points, ambient_dim]
            Vector.
        base_point : array-like, shape=[..., k_sampling_points, ambient_dim]
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
        if shape[-2] != self.k_sampling_points:
            return [False] * shape[0]
        stacked_vec = gs.reshape(vector, (-1, shape[-1]))
        stacked_point = gs.reshape(base_point, (-1, shape[-1]))
        is_tangent = ambient_manifold.is_tangent(stacked_vec, stacked_point, atol)
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
        vector : array-like, shape=[..., k_sampling_points, ambient_dim]
            Vector.
        base_point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.

        Returns
        -------
        tangent_vec : array-like, shape=[..., k_sampling_points, ambient_dim]
            Tangent vector at base point.
        """
        ambient_manifold = self.ambient_manifold
        shape = vector.shape
        stacked_vec = gs.reshape(vector, (-1, shape[-1]))
        stacked_point = gs.reshape(base_point, (-1, shape[-1]))
        tangent_vec = ambient_manifold.to_tangent(stacked_vec, stacked_point)
        tangent_vec = gs.reshape(tangent_vec, vector.shape)
        return tangent_vec

    def projection(self, point):
        """Project a point to the space of discrete curves.

        Parameters
        ----------
        point: array-like, shape[..., k_sampling_points, ambient_dim]
            Point.

        Returns
        -------
        point: array-like, shape[..., k_sampling_points, ambient_dim]
            Point.
        """
        ambient_manifold = self.ambient_manifold
        shape = point.shape
        stacked_point = gs.reshape(point, (-1, shape[-1]))
        projected_point = ambient_manifold.projection(stacked_point)
        projected_point = gs.reshape(projected_point, shape)
        return projected_point

    def random_point(self, n_samples=1, bound=1.0):
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
        k_sampling_points : int
            Number of sampling points for the discrete curves.
            Optional, default : 10.

        Returns
        -------
        samples : array-like, shape=[..., k_sampling_points, {dim, [n, n]}]
            Points sampled on the hypersphere.
        """
        sample = self.ambient_manifold.random_point(n_samples * self.k_sampling_points)
        sample = gs.reshape(sample, (n_samples, self.k_sampling_points, -1))
        return sample[0] if n_samples == 1 else sample


class ClosedDiscreteCurves(LevelSet):
    r"""Space of closed discrete curves sampled at points in ambient_manifold.

    Each individual curve is represented by a 2d-array of shape `[
    k_sampling_points, ambient_dim]`.

    See [Sea2011]_ for details.

    Parameters
    ----------
    ambient_manifold : Manifold
        Manifold in which curves take values.

    Attributes
    ----------
    ambient_manifold : Manifold
        Manifold in which curves take values.
    embedding_space : Manifold
        Manifold in which the space of closed curves is embedded.
    l2_landmarks_metric : callable
        Function that takes as argument an integer number of sampled points
        and returns the corresponding L2 metric (product) metric,
        a RiemannianMetric object
    srv_metric : RiemannianMetric
        Square root velocity metric.

    References
    ----------
    .. [Sea2011] A. Srivastava, E. Klassen, S. H. Joshi and I. H. Jermyn,
        "Shape Analysis of Elastic Curves in Euclidean Spaces,"
        in IEEE Transactions on Pattern Analysis and Machine Intelligence,
        vol. 33, no. 7, pp. 1415-1428, July 2011.
    """

    def __init__(self, ambient_manifold, k_sampling_points=10):
        dim = ambient_manifold.dim * (k_sampling_points - 1)
        super(ClosedDiscreteCurves, self).__init__(
            dim=dim,
            shape=(k_sampling_points,) + ambient_manifold.shape,
            submersion=None,
            tangent_submersion=None,
            value=None,
            embedding_space=DiscreteCurves(
                ambient_manifold=ambient_manifold, k_sampling_points=k_sampling_points
            ),
        )
        self.ambient_manifold = ambient_manifold
        self.ambient_metric = ambient_manifold.metric

    def belongs(self, point, atol=gs.atol):
        """Test whether a point belongs to the manifold.

        Test that all points of the curve belong to the ambient manifold.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points, ambient_dim]
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
        point = gs.to_ndarray(point, to_ndim=3)
        first_point = point[:, 0, :]
        last_point = point[:, -1, :]
        point_belongs = gs.allclose(first_point, last_point, atol=atol)
        point_belongs_to_embedding = self.embedding_space.belongs(point)
        return gs.squeeze(gs.array(point_belongs) and point_belongs_to_embedding)

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at a curve.

        A vector is tangent at a curve if it is a vector field along that
        curve.

        Parameters
        ----------
        vector : array-like, shape=[..., k_sampling_points, ambient_dim]
            Vector.
        base_point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        raise NotImplementedError("The is_tangent method is not implemented.")

    def intrinsic_to_extrinsic_coords(self, point_intrinsic):
        """Convert from intrinsic to extrinsic coordinates.

        Parameters
        ----------
        point_intrinsic : array-like, shape=[..., dim]
            Point in the embedded manifold in intrinsic coordinates.

        Returns
        -------
        point_extrinsic : array-like, shape=[..., dim_embedding]
            Point in the embedded manifold in extrinsic coordinates.
        """
        raise NotImplementedError("intrinsic_to_extrinsic_coords is not implemented.")

    def extrinsic_to_intrinsic_coords(self, point_extrinsic):
        """Convert from extrinsic to intrinsic coordinates.

        Parameters
        ----------
        point_extrinsic : array-like, shape=[..., dim_embedding]
            Point in the embedded manifold in extrinsic coordinates,
            i. e. in the coordinates of the embedding manifold.

        Returns
        -------
        point_intrinsic : array-lie, shape=[..., dim]
            Point in the embedded manifold in intrinsic coordinates.
        """
        raise NotImplementedError("extrinsic_to_intrinsic_coords is not implemented.")

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        As tangent vectors are vector fields along a curve, each component of
        the vector is projected to the tangent space of the corresponding
        point of the discrete curve. The number of sampling points should
        match in the vector and the base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., k_sampling_points, ambient_dim]
            Vector.
        base_point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.

        Returns
        -------
        tangent_vec : array-like, shape=[..., k_sampling_points, ambient_dim]
            Tangent vector at base point.
        """
        raise NotImplementedError("The to_tangent method is not implemented.")

    def random_point(self, n_samples=1):
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
        k_sampling_points : int
            Number of sampling points for the discrete curves.
            Optional, default : 10.

        Returns
        -------
        samples : array-like, shape=[..., k_sampling_points, {dim, [n, n]}]
            Points sampled on the hypersphere.
        """
        sample = self.embedding_space.random_point(n_samples)
        sample = gs.to_ndarray(sample, to_ndim=3)
        sample_minus_last_point = sample[:, :-1, :]
        first_point = gs.reshape(sample[:, 0, :], (sample.shape[0], 1, -1))
        sample = gs.concatenate([sample_minus_last_point, first_point], axis=1)
        return gs.squeeze(sample)

    def projection(self, point, atol=gs.atol, max_iter=1000):
        """Project a discrete curve into the space of closed discrete curves.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.
        atol : float
            Tolerance of the projection algorithm.
            Optional, default: backend atol.
        max_iter : float
            Maximum number of iteration of the algorithm.
            Optional, default: 1000

        Returns
        -------
        proj : array-like, shape=[..., k_sampling_points, ambient_dim]
        """
        is_euclidean = isinstance(self.ambient_manifold, Euclidean)
        is_planar = is_euclidean and self.ambient_manifold.dim == 2

        if not is_planar:
            raise AssertionError(
                "The projection is only implemented "
                "for discrete curves embedded in a "
                "2D Euclidean space."
            )
        point_ndim = point.ndim

        srv_metric = self.embedding_space.srv_metric
        srv = srv_metric.srv_transform(point)
        srv_proj = self.srv_projection(srv, atol=atol, max_iter=max_iter)

        point = gs.to_ndarray(point, to_ndim=3)
        proj = srv_metric.srv_transform_inverse(srv_proj, point[:, 0])
        return proj if point_ndim == 3 else gs.squeeze(proj)

    def srv_projection(self, srv, atol=gs.atol, max_iter=1000):
        """Project a point in the srv space into the space of closed curves srv.

        The algorithm is from the paper cited above and modifies the srv
        iteratively so that G(srv) = (0, ..., 0) with the paper's notation.

        Remark: for now, the algorithm might not converge for some curves such
        as segments.

        Parameters
        ----------
        srv : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.
        atol : float
            Tolerance of the projection algorithm.
            Optional, default: backend atol.
        max_iter : float
            Maximum number of iteration of the algorithm.
            Optional, default: 1000

        Returns
        -------
        proj : array-like, shape=[..., k_sampling_points, ambient_dim]
        """
        is_euclidean = isinstance(self.ambient_metric, EuclideanMetric)
        is_planar = is_euclidean and self.ambient_metric.dim == 2

        if not is_planar:
            raise AssertionError(
                "The projection is only implemented "
                "for discrete curves embedded in a"
                "2D Euclidean space."
            )

        dim = self.ambient_metric.dim
        srv_inner_prod = self.embedding_space.l2_curves_metric.inner_product
        srv_norm = self.embedding_space.l2_curves_metric.norm
        inner_prod = self.ambient_metric.inner_product

        def closeness_criterion(srv, srv_norms):
            """Compute the closeness criterion from [Sea2011]_.

            The closeness criterion is denoted by G(q) in [Sea2011]_, where q
            represents the srv of interest.

            References
            ----------
            .. [Sea2011] A. Srivastava, E. Klassen, S. H. Joshi and I. H. Jermyn,
                "Shape Analysis of Elastic Curves in Euclidean Spaces,"
                in IEEE Transactions on Pattern Analysis and Machine Intelligence,
                vol. 33, no. 7, pp. 1415-1428, July 2011.
            """
            return gs.sum(srv * srv_norms[:, None], axis=0)

        def one_srv_projection(one_srv):
            """Project one srv by iteratively updating until closeness criterion is 0.

            Details can be found in [Sea2011]_ Section 4.2.
            """
            initial_norm = srv_norm(one_srv)
            proj = one_srv
            proj_norms = self.ambient_metric.norm(proj)
            residual = closeness_criterion(proj, proj_norms)
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
                residual = closeness_criterion(proj, proj_norms)
                criteria = self.ambient_metric.norm(residual)

                nb_iter += 1
            return proj

        srv_ndim = srv.ndim
        srv = gs.to_ndarray(srv, to_ndim=3)

        for i_srv, one_srv in enumerate(srv):
            srv[i_srv] = one_srv_projection(one_srv)

        return srv if srv_ndim == 3 else gs.squeeze(srv)


class L2CurvesMetric(RiemannianMetric):
    """L2 metric on the space of discrete curves.

    L2 metric on the space of regularly sampled discrete curves
    defined on the unit interval. The inner product between two tangent vectors
    is given by the integral of the ambient space inner product, approximated by
    a left Riemann sum.
    """

    def __init__(self, ambient_manifold, ambient_metric=None):
        super(L2CurvesMetric, self).__init__(
            dim=math.inf,
            signature=(math.inf, 0, 0),
            shape=(None,) + ambient_manifold.shape,
        )
        if ambient_metric is None:
            if hasattr(ambient_manifold, "metric"):
                self.ambient_metric = ambient_manifold.metric
            else:
                raise ValueError(
                    "Instantiating an object of class "
                    "DiscreteCurves requires either a metric"
                    " or an ambient manifold"
                    " equipped with a metric."
                )
        else:
            self.ambient_metric = ambient_metric
        self.l2_landmarks_metric = lambda n: L2LandmarksMetric(
            ambient_manifold.metric, k_landmarks=n
        )

    @staticmethod
    def riemann_sum(func, missing_last_time=True):
        """Compute the left Riemann sum approximation of the integral.

        Compute the left Riemann sum approximation of the integral of a
        function func defined on on the unit interval,
        given by sample points at regularly spaced times
        t_k = k / k_sampling_points for k = 0, ..., k_sampling_points.

        Parameters
        ----------
        func : array-like, shape=[..., k_sampling_points]
            Sample points of a function at regularly spaced times.
        missing_last_time : boolean.
            Is true when the last value at to time 1 is missing.
            Optional, default True.

        Returns
        -------
        riemann_sum : array-like, shape=[..., ]
            Left Riemann sum.
        """
        func = gs.to_ndarray(func, to_ndim=2)
        k_sampling_points = func.shape[-1] + 1 if missing_last_time else func.shape[-1]
        dt = 1 / k_sampling_points
        values_to_sum = func if missing_last_time else func[:, :-1]
        riemann_sum = dt * gs.sum(values_to_sum, axis=-1)
        return gs.squeeze(riemann_sum)

    def pointwise_inner_products(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the pointwise inner products of a pair of tangent vectors.

        Compute the inner-products between the components of two tangent vectors
        at the different sampling points of a base curve.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., k_sampling_points, ambient_dim]
            Tangent vector to discrete curve.
        tangent_vec_b : array-like, shape=[..., k_sampling_points, ambient_dim]
            Tangent vector to discrete curve.
        base_point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Point representing a discrete curve.
            Optional, default None.

        Returns
        -------
        inner_prod : array-like, shape=[..., k_sampling_points]
            Point-wise inner-product.
        """

        def inner_prod_aux(vec_a, vec_b, curve=None):
            """Compute inner-product of the ambient metric."""
            inner_prod = self.ambient_metric.inner_product(vec_a, vec_b, curve)
            return gs.squeeze(inner_prod)

        if base_point is None:
            return gs.vectorize(
                (tangent_vec_a, tangent_vec_b),
                inner_prod_aux,
                dtype=tangent_vec_a.dtype,
                multiple_args=True,
                signature="(i,j),(i,j)->(i)",
            )

        return gs.vectorize(
            (tangent_vec_a, tangent_vec_b, base_point),
            inner_prod_aux,
            dtype=tangent_vec_a.dtype,
            multiple_args=True,
            signature="(i,j),(i,j),(i,j)->(i)",
        )

    def pointwise_norms(self, tangent_vec, base_point=None):
        """Compute the pointwise norms of a tangent vector.

        Compute the norms of the components of a tangent vector at the different
        sampling points of a base curve.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., k_sampling_points, ambient_dim]
            Tangent vector to discrete curve.
        base_point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Point representing a discrete curve.

        Returns
        -------
        norm : array-like, shape=[..., k_sampling_points]
            Point-wise norms.
        """
        sq_norm = self.pointwise_inner_products(
            tangent_vec_a=tangent_vec, tangent_vec_b=tangent_vec, base_point=base_point
        )
        return gs.sqrt(sq_norm)

    def inner_product(
        self, tangent_vec_a, tangent_vec_b, base_point=None, missing_last_time=True
    ):
        """Compute L2 inner product between two tangent vectors.

        The inner product is the integral of the ambient space inner product,
        approximated by a left Riemann sum.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., k_sampling_points, ambient_dim]
            Tangent vector to a curve, i.e. infinitesimal vector field
            along a curve.
        tangent_vec_b : array-like, shape=[..., k_sampling_points, ambient_dim]
            Tangent vector to a curve, i.e. infinitesimal vector field
            along a curve.
        base_point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve defined on the unit interval [0, 1].
        missing_last_time : boolean.
            Is true when the values of the tangent vectors at time 1 are missing.
            Optional, default True.

        Return
        ------
        inner_prod : array_like, shape=[...]
            L2 inner product between tangent_vec_a and tangent_vec_b.
        """
        inner_products = self.pointwise_inner_products(
            tangent_vec_a, tangent_vec_b, base_point
        )
        return self.riemann_sum(inner_products, missing_last_time)

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute Riemannian exponential of tangent vector wrt to base curve.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., k_sampling_points, ambient_dim]
            Tangent vector to discrete curve.
        base_point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.

        Return
        ------
        end_curve :  array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve, result of the Riemannian exponential.
        """
        k_sampling_points = base_point.shape[-2]
        l2_landmarks_metric = self.l2_landmarks_metric(k_sampling_points)
        return l2_landmarks_metric.exp(tangent_vec, base_point)

    def log(self, point, base_point, **kwargs):
        """Compute Riemannian logarithm of a curve wrt a base curve.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.
        base_point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve to use as base point.

        Returns
        -------
        log : array-like, shape=[..., k_sampling_points, ambient_dim]
            Tangent vector to a discrete curve.
        """
        k_sampling_points = base_point.shape[-2]
        l2_landmarks_metric = self.l2_landmarks_metric(k_sampling_points)
        return l2_landmarks_metric.log(point, base_point)

    def geodesic(self, initial_point, end_point=None, initial_tangent_vec=None):
        """Compute geodesic from initial curve to end curve.

        Geodesic specified either by an initial curve and an end curve,
        either by an initial curve and an initial tangent vector.

        Parameters
        ----------
        initial_point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.
        end_point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve. If None, an initial tangent vector must be given.
            Optional, default : None
        initial_tangent_vec : array-like,
            shape=[..., k_sampling_points, ambient_dim]
            Tangent vector at base curve, the initial speed of the geodesics.
            If None, an end curve must be given and a logarithm is computed.
            Optional, default : None

        Returns
        -------
        geodesic : callable
            The time parameterized geodesic curve.
        """
        k_sampling_points = initial_point.shape[-2]
        l2_landmarks_metric = self.l2_landmarks_metric(k_sampling_points)
        return l2_landmarks_metric.geodesic(
            initial_point, end_point, initial_tangent_vec
        )


class ElasticMetric(RiemannianMetric):
    """Elastic metric defined using the F_transform.

    Each individual curve is represented by a 2d-array of shape `[
    k_sampling_points, ambient_dim]`.

    See [NK2018]_ for details.

    Parameters
    ----------
    ambient_manifold : Manifold
        Manifold in which curves take values.
    metric : RiemannianMetric
        Metric to use on the ambient manifold. If None is passed, ambient
        manifold should have a metric attribute, which will be used.
        Optional, default : None.
    translation_invariant : bool
        Optional, default : True.
    a : float
        Bending parameter.
    b : float
        Stretching parameter.

    References
    ----------
    .. [KN2018] S. Kurtek and T. Needham,
        "Simplifying transforms for general elastic metrics on the space of
        plane curves", arXiv:1803.10894 [math.DG], 29 Mar 2018.
    """

    def __init__(
        self, a, b, ambient_manifold=R2, ambient_metric=None, translation_invariant=True
    ):
        super(ElasticMetric, self).__init__(
            dim=math.inf,
            signature=(math.inf, 0, 0),
            shape=(None,) + ambient_manifold.shape,
        )
        self.ambient_metric = ambient_metric
        if ambient_metric is None:
            if hasattr(ambient_manifold, "metric"):
                self.ambient_metric = ambient_manifold.metric
            else:
                raise ValueError(
                    "Instantiating an object of class "
                    "ElasticMetric requires either a metric"
                    " or an ambient manifold"
                    " equipped with a metric."
                )
        self.ambient_manifold = ambient_manifold
        self.l2_curves_metric = L2CurvesMetric(ambient_manifold=ambient_manifold)
        self.translation_invariant = translation_invariant
        self.a = a
        self.b = b

    def cartesian_to_polar(self, tangent_vec):
        """Compute polar coordinates of a tangent vector from the cartesian ones.

        This function is an auxiliary function used for the computation
        of the f_transform and its inverse, and is applied to the derivative
        of a curve.

        See [KN2018]_ for details.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., k_sampling_points, ambient_dim]
            Tangent vector, representing the derivative c' of a discrete curve c.

        Returns
        -------
        norms : array-like, shape=[..., k_sampling_points]
            Norms of the components of the tangent vector in polar coordinates.
        args : array-like, shape=[..., k_sampling_points]
            Arguments, i.e. angle, of the components in polar coordinates.
        """
        if not (
            isinstance(self.ambient_manifold, Euclidean)
            and self.ambient_manifold.dim == 2
        ):
            raise NotImplementedError(
                "cartesian_to_polar is only implemented for planar curves:\n"
                "ambient_manifold must be a plane, but it is:\n"
                f"{self.ambient_manifold} of dimension {self.ambient_manifold.dim}."
            )
        k_sampling_points = tangent_vec.shape[-2]
        inner_prod = self.ambient_metric.inner_product

        norms = self.ambient_metric.norm(tangent_vec)
        arg_0 = gs.arctan2(tangent_vec[..., 0, 1], tangent_vec[..., 0, 0])
        args = [arg_0]

        for i in range(1, k_sampling_points):
            point, last_point = tangent_vec[..., i, :], tangent_vec[..., i - 1, :]
            prod = inner_prod(point, last_point)
            cosine = prod / (norms[..., i] * norms[..., i - 1])
            angle = gs.arccos(gs.clip(cosine, -1, 1))
            det = gs.linalg.det(gs.stack([last_point, point], axis=-1))
            orientation = gs.sign(det)
            arg = args[-1] + orientation * angle
            args.append(arg)

        args = gs.stack(args, axis=-1)
        polar_tangent_vec = gs.stack([norms, args], axis=-1)

        return polar_tangent_vec

    def polar_to_cartesian(self, polar_tangent_vec):
        """Compute the cartesian coordinates of a tangent vector from polar ones.

        Parameters
        ----------
        norms : array-like, shape=[..., k_sampling_points]
            Norms of the components.
        args : array-like, shape=[..., k_sampling_points]
            Arguments of the components.

        Returns
        -------
        tangent_vec : array-like, shape=[..., k_sampling_points, ambient_dim]
            Tangent vector.
        """
        if not (
            isinstance(self.ambient_manifold, Euclidean)
            and self.ambient_manifold.dim == 2
        ):
            raise NotImplementedError(
                "polar_to_cartesian is only implemented for planar curves:\n"
                "ambient_manifold must be a plane, but it is:\n"
                f"{self.ambient_manifold} of dimension {self.ambient_manifold.dim}."
            )
        tangent_vec_x = gs.cos(polar_tangent_vec[..., :, 1])
        tangent_vec_y = gs.sin(polar_tangent_vec[..., :, 1])
        norms = polar_tangent_vec[..., :, 0]
        unit_tangent_vec = gs.stack((tangent_vec_x, tangent_vec_y), axis=-1)
        tangent_vec = norms[..., :, None] * unit_tangent_vec

        return tangent_vec

    def f_transform(self, point):
        r"""Compute the f_transform of a curve.

        Note that the f_transform is defined on the space of curves
        quotiented by translations, which is identified with the space
        of curves with their first sampling point located at 0:

        .. math::
            curve(0) = (0, 0)

        The f_transform is given by the formula:

        .. math::
            Imm(I, R^2) / R^2 \mapsto C^\infty(I, C*)
            c \mapsto 2b |c'|^{1/2} (\frac{c'}{|c'|})^{a/(2b)}

        where the identification :math:`C = R^2` is used and
        the exponentiation is a complex exponentiation, which can make
        the f_transform not well-defined:

        .. math::
            f(c) = 2b r^{1/2}\exp(i\theta * a/(2b)) * \exp(ik\pi * a/b)

         where (r, theta) is the polar representation of c', and for
         any :math:`k \in Z`.

        The implementation uses formula (3) from [KN2018]_ , i.e. choses
        the representative corresponding to k = 0.

        Notes
        -----
        f_transform is a bijection if and only if a/2b=1.

        If a 2b is an integer not equal to 1:
          - then f_transform is well-defined but many-to-one.

        If a 2b is not an integer:
          - then f_transform is multivalued,
          - and f_transform takes finitely many values if and only if a 2b is rational.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.

        Returns
        -------
        f : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            F_transform of the curve..
        """
        if not (
            isinstance(self.ambient_manifold, Euclidean)
            and self.ambient_manifold.dim == 2
        ):
            raise NotImplementedError(
                "f_transform is only implemented for planar curves:\n"
                "ambient_manifold must be a plane, but it is:\n"
                f"{self.ambient_manifold} of dimension {self.ambient_manifold.dim}."
            )
        k_sampling_points = point.shape[-2]
        velocity = (k_sampling_points - 1) * (point[..., 1:, :] - point[..., :-1, :])
        polar_velocity = self.cartesian_to_polar(velocity)
        speeds = polar_velocity[..., :, 0]
        args = polar_velocity[..., :, 1]

        f_args = args * self.a / (2 * self.b)
        f_norms = 2 * self.b * gs.sqrt(speeds)
        f_polar = gs.stack([f_norms, f_args], axis=-1)
        f_cartesian = self.polar_to_cartesian(f_polar)

        return f_cartesian

    def f_transform_inverse(self, f_trans, starting_sampling_point):
        r"""Compute the inverse F_transform of a transformed curve.

        This only works if a / (2b) <= 1.
        See [KN2018]_ for details.

        When the f_transform is many-to-one, one antecedent is chosen.

        Notes
        -----
        f_transform is a bijection if and only if a / (2b) = 1.

        If a / (2b) is an integer not equal to 1:
          - then f_transform is well-defined but many-to-one.

        If a / (2b) is not an integer:
          - then f_transform is multivalued,
          - and f_transform takes finitely many values if and only if a 2b is rational.

        Parameters
        ----------
        f_trans : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            f-transform of a discrete curve.

        starting_sampling_point: array-like, shape=[..., ambient_dim]
            Point of the ambient manifold to use as start of the retrieved curve.

        Returns
        -------
        curve : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.
        """
        if not (
            isinstance(self.ambient_manifold, Euclidean)
            and self.ambient_manifold.dim == 2
        ):
            raise NotImplementedError(
                "f_transform_inverse is only implemented for planar curves:\n"
                "ambient_manifold must be a plane, but it is:\n"
                f"{self.ambient_manifold} of dimension {self.ambient_manifold.dim}."
            )

        if self.a / (2 * self.b) > 1:
            raise NotImplementedError(
                "f_transform_inverse is only implemented for a / (2b) <= 1."
            )
        if gs.ndim(f_trans) != gs.ndim(starting_sampling_point):
            starting_sampling_point = gs.to_ndarray(
                starting_sampling_point, to_ndim=f_trans.ndim, axis=-2
            )

        k_sampling_points_minus_one = f_trans.shape[-2]

        f_polar = self.cartesian_to_polar(f_trans)
        f_norms = f_polar[..., :, 0]
        f_args = f_polar[..., :, 1]

        dt = 1 / k_sampling_points_minus_one

        delta_points_x = gs.einsum(
            "...i,...i->...i", dt * f_norms**2, gs.cos(2 * self.b / self.a * f_args)
        )
        delta_points_y = gs.einsum(
            "...i,...i->...i", dt * f_norms**2, gs.sin(2 * self.b / self.a * f_args)
        )

        delta_points = gs.stack((delta_points_x, delta_points_y), axis=-1)

        delta_points = 1 / (4 * self.b**2) * delta_points

        curve = gs.concatenate([starting_sampling_point, delta_points], axis=-2)
        curve = gs.cumsum(curve, -2)
        return gs.squeeze(curve)

    def dist(self, point_a, point_b, rescaled=False):
        """Compute the geodesic distance between two curves.

        The two F_transforms are computed with corrected arguments
        before taking the L2 distance between them.
        See [KN2018]_ for details.

        Parameters
        ----------
        point_a : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.
        point_b : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.

        Returns
        -------
        dist : [...]
            Geodesic distance between the curves.
        """
        if not (
            isinstance(self.ambient_manifold, Euclidean)
            and self.ambient_manifold.dim == 2
        ):
            raise NotImplementedError(
                "dist is only implemented for planar curves:\n"
                "ambient_manifold must be a plane, but it is:\n"
                f"{self.ambient_manifold} of dimension {self.ambient_manifold.dim}."
            )

        f_1 = self.f_transform(point_a)
        f_2 = self.f_transform(point_b)

        if rescaled:
            cosine = self.l2_curves_metric.inner_product(f_1, f_2) / (4 * self.b**2)
            distance = 2 * self.b * gs.arccos(gs.clip(cosine, -1, 1))
        else:
            distance = self.l2_curves_metric.dist(f_1, f_2)

        return distance

    def squared_dist(self, point_a, point_b, rescaled=False):
        """Compute squared geodesic distance between two curves.

        The two F_transforms are computed with corrected arguments
        before taking the L2 distance between them.
        See [KN2018]_ for details.

        Parameters
        ----------
        point_a : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.
        point_b : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.

        Returns
        -------
        _ : [...]
            Squared geodesic distance between the curves.
        """
        return self.dist(point_a=point_a, point_b=point_b, rescaled=rescaled) ** 2

    def geodesic(self, initial_point, end_point=None, initial_tangent_vec=None):
        """Compute geodesic from initial curve to end curve.

        Geodesic specified by an initial curve and an end curve computed
        as an inverse f_transform of a segment between f_transforms.

        Parameters
        ----------
        initial_point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.
        end_point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.

        Returns
        -------
        curve_on_geodesic : callable
            The time parameterized geodesic curve.
        """
        if not isinstance(self.ambient_metric, EuclideanMetric):
            raise AssertionError(
                "The geodesics are only implemented for "
                "discrete curves embedded in a "
                "Euclidean space."
            )
        curve_ndim = 2
        initial_point = gs.to_ndarray(initial_point, to_ndim=curve_ndim)
        end_point = gs.to_ndarray(end_point, to_ndim=curve_ndim)

        def path(times):
            """Generate parametrized function for geodesic.

            Parameters
            ----------
            times : array-like, shape=[n_times,]
                Times in [0, 1] at which to compute points of the geodesic.
            """
            times = gs.to_ndarray(times, to_ndim=1)

            curves_path = []
            for t in times:
                initial_f = self.f_transform(initial_point)
                end_f = self.f_transform(end_point)
                f_t = (1 - t) * initial_f + t * end_f
                curve_t = self.f_transform_inverse(
                    f_t, gs.zeros(curve_ndim, dtype=initial_point.dtype)
                )
                curves_path.append(curve_t)
            return gs.stack(curves_path)

        return path


class SRVMetric(ElasticMetric):
    """Elastic metric defined using the Square Root Velocity Function.

    The SRV metric is equivalent to the elastic metric chosen with
    - bending parameter a = 1,
    - stretching parameter b = 1/2.

    See [Sea2011]_ for details.

    Parameters
    ----------
    ambient_manifold : Manifold
        Manifold in which curves take values.
    metric : RiemannianMetric
        Metric to use on the ambient manifold. If None is passed, ambient
        manifold should have a metric attribute, which will be used.
        Optional, default : None.
    translation_invariant : bool
        Optional, default : True.

    References
    ----------
    .. [Sea2011] A. Srivastava, E. Klassen, S. H. Joshi and I. H. Jermyn,
        "Shape Analysis of Elastic Curves in Euclidean Spaces,"
        in IEEE Transactions on Pattern Analysis and Machine Intelligence,
        vol. 33, no. 7, pp. 1415-1428, July 2011.
    """

    def __init__(
        self, ambient_manifold, ambient_metric=None, translation_invariant=True
    ):
        super(SRVMetric, self).__init__(
            a=1,
            b=0.5,
            ambient_manifold=ambient_manifold,
            ambient_metric=ambient_metric,
            translation_invariant=translation_invariant,
        )

    def srv_transform(self, point, tol=gs.atol):
        r"""Square Root Velocity Transform (SRVT).

        Compute the square root velocity representation of a curve. The
        velocity is computed using the log map.

        In the case of several curves, an index selection procedure allows to
        get rid of the log between the end point of curve[k, :, :] and the starting
        point of curve[k + 1, :, :].

        .. math::
            c \mapsto \frac{c'}{|c'|^{1/2}}

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.

        tol : float
            Tolerance value to decide duplicity of two consecutive sample
            points on a given Discrete Curve.

        Returns
        -------
        srv : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Square-root velocity representation of a discrete curve.
        """
        if gs.any(
            self.ambient_metric.norm(point[..., 1:, :] - point[..., :-1, :]) < tol
        ):
            raise AssertionError(
                "The square root velocity framework "
                "is only defined for discrete curves "
                "with distinct consecutive sample points."
            )
        point_ndim = gs.ndim(point)
        point = gs.to_ndarray(point, to_ndim=3)
        n_points, k_sampling_points, n_coords = point.shape
        srv_shape = (n_points, k_sampling_points - 1, n_coords)

        point = gs.reshape(point, (n_points * k_sampling_points, n_coords))
        coef = k_sampling_points - 1
        velocity = coef * self.ambient_metric.log(
            point=point[1:, :], base_point=point[:-1, :]
        )
        velocity_norm = self.ambient_metric.norm(velocity, point[:-1, :])
        srv = gs.einsum("...i,...->...i", velocity, 1.0 / gs.sqrt(velocity_norm))

        index = gs.arange(n_points * k_sampling_points - 1)
        mask = ~((index + 1) % k_sampling_points == 0)
        srv = gs.reshape(srv[mask], srv_shape)

        if point_ndim == 2:
            return gs.squeeze(srv)
        return srv

    def f_transform(self, point):
        """Compute the F_transform of a curve.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.

        Returns
        -------
        f : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            F_transform of the curve.
        """
        return self.srv_transform(point)

    def srv_transform_inverse(self, srv, starting_sampling_point):
        r"""Inverse of the Square Root Velocity Transform (SRVT).

        Retrieve a curve from its square root velocity representation and
        starting point.

        .. math::
            c(t) = c(0) + \int_0^t q(s) |q(s)|ds

        with:
        - c the curve that can be retrieved only up to a translation,
        - q the srv representation of the curve,
        - c(0) the starting point of the curve.


        See [Sea2011]_ Section 2.1 for details.

        Parameters
        ----------
        srv : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Square-root velocity representation of a discrete curve.
        starting_sampling_point : array-like, shape=[..., ambient_dim]
            Point of the ambient manifold to use as start of the retrieved curve.

        Returns
        -------
        curve : array-like, shape=[..., k_sampling_points, ambient_dim]
            Curve retrieved from its square-root velocity.
        """
        if not isinstance(self.ambient_metric, EuclideanMetric):
            raise AssertionError(
                "The square root velocity inverse is only "
                "implemented for discrete curves embedded "
                "in a Euclidean space."
            )
        if gs.ndim(srv) != gs.ndim(starting_sampling_point):
            starting_sampling_point = gs.to_ndarray(
                starting_sampling_point, to_ndim=srv.ndim, axis=-2
            )
        srv_shape = srv.shape
        srv = gs.to_ndarray(srv, to_ndim=3)
        n_curves, k_sampling_points_minus_one, n_coords = srv.shape

        srv_flat = gs.reshape(srv, (n_curves * k_sampling_points_minus_one, n_coords))
        srv_norm = self.ambient_metric.norm(srv_flat)

        dt = 1 / k_sampling_points_minus_one

        delta_points = gs.einsum("...,...i->...i", dt * srv_norm, srv_flat)
        delta_points = gs.reshape(delta_points, srv_shape)

        curve = gs.concatenate((starting_sampling_point, delta_points), -2)

        curve = gs.cumsum(curve, -2)

        return curve

    def f_transform_inverse(self, f_trans, starting_sampling_point):
        """Compute the inverse of the F_transform of a transformed curve.

        See [KN2018]_ for details.

        Parameters
        ----------
        f_trans : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            f-transform of a discrete curve.

        starting_sampling_point: array-like, shape=[..., ambient_dim]
            Point of the ambient manifold to use as start of the retrieved curve.

        Returns
        -------
        curve : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.
        """
        return self.srv_transform_inverse(f_trans, starting_sampling_point)

    def aux_differential_srv_transform(self, tangent_vec, point):
        """Compute differential of the square root velocity transform.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., k_sampling_points, ambient_dim]
            Tangent vector to curve, i.e. infinitesimal vector field
            along curve.
        point : array-like, shape=[..., k_sampling_points, ambiend_dim]
            Discrete curve.

        Returns
        -------
        d_srv_vec : array-like, shape=[..., ambient_dim,]
            Differential of the square root velocity transform at curve
            evaluated at tangent_vec.
        """
        if not isinstance(self.ambient_metric, EuclideanMetric):
            raise AssertionError(
                "The differential of the square root "
                "velocity function is only implemented for "
                "discrete curves embedded in a Euclidean "
                "space."
            )
        k_sampling_points = point.shape[-2]
        d_vec = (k_sampling_points - 1) * (
            tangent_vec[..., 1:, :] - tangent_vec[..., :-1, :]
        )
        velocity_vec = (k_sampling_points - 1) * (
            point[..., 1:, :] - point[..., :-1, :]
        )
        velocity_norm = self.ambient_metric.norm(velocity_vec)
        unit_velocity_vec = gs.einsum(
            "...ij,...i->...ij", velocity_vec, 1 / velocity_norm
        )

        inner_prod = self.l2_curves_metric.pointwise_inner_products(
            d_vec, unit_velocity_vec, point[..., :-1, :]
        )
        d_vec_tangential = gs.einsum("...ij,...i->...ij", unit_velocity_vec, inner_prod)
        d_srv_vec = d_vec - 1 / 2 * d_vec_tangential
        d_srv_vec = gs.einsum(
            "...ij,...i->...ij", d_srv_vec, 1 / velocity_norm ** (1 / 2)
        )

        return d_srv_vec

    def aux_differential_srv_transform_inverse(self, tangent_vec, point):
        """Compute inverse of differential of the square root velocity transform.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Tangent vector to srv.
        point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.

        Returns
        -------
        vec : array-like, shape=[..., ambient_dim]
            Inverse of the differential of the square root velocity transform at
            curve evaluated at tangent_vec.
        """
        if not isinstance(self.ambient_metric, EuclideanMetric):
            raise AssertionError(
                "The differential of the square root "
                "velocity function is only implemented for "
                "discrete curves embedded in a Euclidean "
                "space."
            )

        point = gs.to_ndarray(point, to_ndim=3)
        n_points, k_sampling_points, ambient_dim = point.shape

        k_sampling_points = point.shape[-2]
        velocity_vec = (k_sampling_points - 1) * (
            point[..., 1:, :] - point[..., :-1, :]
        )
        velocity_norm = self.ambient_metric.norm(velocity_vec)
        unit_velocity_vec = gs.einsum(
            "...ij,...i->...ij", velocity_vec, 1 / velocity_norm
        )
        inner_prod = self.l2_curves_metric.pointwise_inner_products(
            tangent_vec, unit_velocity_vec, point[..., :-1, :]
        )
        tangent_vec_tangential = gs.einsum(
            "...ij,...i->...ij", unit_velocity_vec, inner_prod
        )
        d_vec = tangent_vec + tangent_vec_tangential
        d_vec = gs.einsum("...ij,...i->...ij", d_vec, velocity_norm ** (1 / 2))
        increment = d_vec / (k_sampling_points - 1)
        initial_value = gs.zeros((n_points, 1, ambient_dim))

        n_increments, _, _ = increment.shape
        if n_points != n_increments:
            if n_points == 1:
                initial_value = gs.tile(initial_value, (n_increments, 1, 1))
            elif n_increments == 1:
                increment = gs.tile(increment, (n_points, 1, 1))
            else:
                raise ValueError("Number of curves and of increments are incompatible.")

        vec = gs.concatenate((initial_value, increment), -2)

        vec = gs.cumsum(vec, -2)

        return gs.squeeze(vec)

    def inner_product(self, tangent_vec_a, tangent_vec_b, point):
        """Compute inner product between two tangent vectors.

        The SRV metric is used, and is computed as pullback of the
        L2 metric by the square root velocity transform.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., k_sampling_points, ambient_dim]
            Tangent vector to curve, i.e. infinitesimal vector field
            along curve.
        tangent_vec_b : array-like, shape=[..., k_sampling_points, ambient_dim]
            Tangent vector to curve, i.e. infinitesimal vector field
        point : array-like, shape=[..., k_sampling_points, ambiend_dim]
            Discrete curve.

        Return
        ------
        inner_prod : array_like, shape=[...]
            Square root velocity inner product between tangent_vec_a and
            tangent_vec_b.
        """
        if not isinstance(self.ambient_metric, EuclideanMetric):
            raise AssertionError(
                "The square root velocity inner product "
                "is only implemented for discrete curves "
                "embedded in a Euclidean space."
            )
        d_srv_vec_a = self.aux_differential_srv_transform(tangent_vec_a, point)
        d_srv_vec_b = self.aux_differential_srv_transform(tangent_vec_b, point)
        inner_prod = self.l2_curves_metric.inner_product(d_srv_vec_a, d_srv_vec_b)

        if not self.translation_invariant:
            inner_prod += self.ambient_metric.inner_product(
                tangent_vec_a[0], tangent_vec_b[0]
            )
        return inner_prod

    def exp(self, tangent_vec, base_point):
        """Compute Riemannian exponential of tangent vector wrt to base curve.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., k_sampling_points, ambient_dim]
            Tangent vector to discrete curve.
        base_point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.

        Return
        ------
        end_curve :  array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve, result of the Riemannian exponential.
        """
        if not isinstance(self.ambient_metric, EuclideanMetric):
            raise AssertionError(
                "The exponential map is only implemented "
                "for discrete curves embedded in a "
                "Euclidean space."
            )
        base_point = gs.to_ndarray(base_point, to_ndim=3)
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)

        base_point, tangent_vec = gs.broadcast_arrays(base_point, tangent_vec)

        base_curve_srv = self.srv_transform(base_point)

        d_srv_tangent_vec = self.aux_differential_srv_transform(
            tangent_vec=tangent_vec, point=base_point
        )
        end_curve_srv = self.l2_curves_metric.exp(
            tangent_vec=d_srv_tangent_vec, base_point=base_curve_srv
        )
        end_curve_starting_point = self.ambient_metric.exp(
            tangent_vec=tangent_vec[:, 0, :], base_point=base_point[:, 0, :]
        )
        end_curve = self.srv_transform_inverse(end_curve_srv, end_curve_starting_point)

        return end_curve

    def log(self, point, base_point):
        """Compute Riemannian logarithm of a curve wrt a base curve.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.
        base_point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve to use as base point.

        Returns
        -------
        log : array-like, shape=[..., k_sampling_points, ambient_dim]
            Tangent vector to a discrete curve.
        """
        if not isinstance(self.ambient_metric, EuclideanMetric):
            raise AssertionError(
                "The logarithm map is only implemented "
                "for discrete curves embedded in a "
                "Euclidean space."
            )

        point = gs.to_ndarray(point, to_ndim=3)
        base_point = gs.to_ndarray(base_point, to_ndim=3)

        point, base_point = gs.broadcast_arrays(point, base_point)

        curve_srv = self.srv_transform(point)
        base_curve_srv = self.srv_transform(base_point)
        log = self.aux_differential_srv_transform_inverse(
            curve_srv - base_curve_srv, base_point
        )
        log = gs.to_ndarray(log, to_ndim=3)

        log_starting_points = self.ambient_metric.log(
            point=point[:, 0, :], base_point=base_point[:, 0, :]
        )
        log_starting_points = gs.to_ndarray(log_starting_points, to_ndim=3, axis=1)
        log += log_starting_points
        return gs.squeeze(log)

    def geodesic(self, initial_point, end_point=None, initial_tangent_vec=None):
        """Compute geodesic from initial curve to end curve.

        Geodesic specified either by an initial curve and an end curve,
        either by an initial curve and an initial tangent vector.

        Parameters
        ----------
        initial_point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.
        end_point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve. If None, an initial tangent vector must be given.
            Optional, default : None
        initial_tangent_vec : array-like,
            shape=[..., k_sampling_points, ambient_dim]
            Tangent vector at base curve, the initial speed of the geodesics.
            If None, an end curve must be given and a logarithm is computed.
            Optional, default : None

        Returns
        -------
        geodesic : callable
            Geodesic in the space of discrete curves.
        """
        if not isinstance(self.ambient_metric, EuclideanMetric):
            raise AssertionError(
                "The geodesics are only implemented for "
                "discrete curves embedded in a "
                "Euclidean space."
            )
        point_ndim = 2
        initial_point = gs.to_ndarray(initial_point, to_ndim=point_ndim + 1)

        if end_point is None and initial_tangent_vec is None:
            raise ValueError(
                "Specify an end curve or an initial tangent "
                "vector to define the geodesic."
            )
        if end_point is not None:
            end_point = gs.to_ndarray(end_point, to_ndim=point_ndim + 1)
            shooting_tangent_vec = self.log(point=end_point, base_point=initial_point)
            if (initial_tangent_vec is not None) and (
                not gs.allclose(shooting_tangent_vec, initial_tangent_vec)
            ):
                raise RuntimeError(
                    "The shooting tangent vector is too"
                    " far from the initial tangent vector."
                )
            initial_tangent_vec = shooting_tangent_vec
        initial_tangent_vec = gs.array(initial_tangent_vec)
        initial_tangent_vec = gs.to_ndarray(initial_tangent_vec, to_ndim=point_ndim + 1)

        def path(t):
            """Generate parametrized function for geodesic.

            Parameters
            ----------
            t: array-like, shape=[n_points,]
                Times at which to compute points of the geodesic.
            """
            t = gs.cast(t, initial_point.dtype)
            t = gs.to_ndarray(t, to_ndim=1)
            t = gs.to_ndarray(t, to_ndim=2, axis=1)
            new_initial_point = gs.to_ndarray(initial_point, to_ndim=point_ndim + 1)
            new_initial_tangent_vec = gs.to_ndarray(
                initial_tangent_vec, to_ndim=point_ndim + 1
            )

            tangent_vecs = gs.einsum("il,nkm->ikm", t, new_initial_tangent_vec)

            point_at_time_t = []
            for tan_vec in tangent_vecs:
                point_at_time_t.append(self.exp(tan_vec, new_initial_point))
            return gs.stack(point_at_time_t)

        return path

    def dist(self, point_a, point_b, **kwargs):
        """Geodesic distance between two curves.

        Parameters
        ----------
        point_a : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.
        point_b : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.

        Returns
        -------
        dist : array-like, shape=[...,]
        """
        if not isinstance(self.ambient_metric, EuclideanMetric):
            raise AssertionError(
                "The distance is only implemented for "
                "discrete curves embedded in a "
                "Euclidean space."
            )
        if point_a.shape != point_b.shape:
            raise ValueError("The curves need to have the same shapes.")

        srv_a = self.srv_transform(point_a)
        srv_b = self.srv_transform(point_b)
        dist_starting_points = self.ambient_metric.dist(point_a[0, :], point_b[0, :])
        dist_srvs = self.l2_curves_metric.norm(srv_b - srv_a)
        if self.translation_invariant:
            return dist_srvs

        dist = gs.sqrt(dist_starting_points**2 + dist_srvs**2)
        return dist

    @staticmethod
    def space_derivative(curve):
        """Compute space derivative of curve using centered differences.

        Parameters
        ----------
        curve : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve.

        Returns
        -------
        space_deriv : array-like, shape=[...,k_sampling_points, ambient_dim]
        """
        n_points = curve.shape[-2]
        if n_points < 2:
            raise ValueError("The curve needs to have at least 2 points.")

        vec_1 = gs.array([-1.0] + [0.0] * (n_points - 2) + [1.0])
        vec_2 = gs.array([1.0 / 2] * (n_points - 2) + [1.0])
        vec_3 = gs.array([1.0] + [1.0 / 2] * (n_points - 2))

        mat_1 = from_vector_to_diagonal_matrix(vec_1, 0)
        mat_2 = from_vector_to_diagonal_matrix(vec_2, -1)
        mat_3 = from_vector_to_diagonal_matrix(vec_3, 1)
        mat_space_deriv = mat_1 - mat_2 + mat_3

        space_deriv = n_points * gs.matmul(mat_space_deriv, curve)

        return space_deriv


class SRVShapeBundle(DiscreteCurves, FiberBundle):
    """Principal bundle of shapes of curves induced by the SRV metric.

    The space of parameterized curves is the total space of a principal
    bundle where the group action is given by reparameterization and the
    base space is the shape space of curves modulo reparametrization, i.e.
    unparametrized curves. In the discrete case, reparametrization corresponds
    to resampling.

    Each tangent vector to the space of parameterized curves can be
    split into a vertical part (tangent to the fibers of the principal
    bundle) and a horizontal part (orthogonal to the vertical part with
    respect to the SRV metric). Horizontal geodesics in the total space
    can be computed using an algorithm that iteratively finds the best
    correspondence between two fibers of the principal bundle, see Reference
    below.

    References
    ----------
    .. [LAB2017] A. Le Brigant,
        "A discrete framework to find the optimal matching between manifold-
        valued curves," in Journal of Mathematical Imaging and Vision, 61,
        pp. 40-70, 2019.
    """

    def __init__(self, ambient_manifold, k_sampling_points=10):
        super(SRVShapeBundle, self).__init__(
            ambient_manifold=ambient_manifold,
            k_sampling_points=k_sampling_points,
        )
        self.total_space_metric = SRVMetric(ambient_manifold=ambient_manifold)
        self.l2_curves_metric = L2CurvesMetric(ambient_manifold=ambient_manifold)

    def vertical_projection(self, tangent_vec, point, return_norm=False):
        """Compute vertical part of tangent vector at base point.

        Parameters
        ----------
        tangent_vec : array-like,
            shape=[..., k_sampling_points, ambient_dim]
            Tangent vector to decompose into horizontal and vertical parts.
        point : array-like,
            shape=[..., k_sampling_points, ambient_dim]
            Discrete curve, base point of tangent_vec in the manifold of curves.
        return_norm : boolean,
            If True, the method returns the pointwise norm of the vertical
            part of tangent_vec.
            Optional, default is False.

        Returns
        -------
        tangent_vec_ver : array-like,
            shape=[..., k_sampling_points, ambient_dim]
            Vertical part of tangent_vec.
        vertical_norm: array-like, shape=[..., n_points]
            Pointwise norm of the vertical part of tangent_vec.
            Only returned when return_norm is True.
        """
        ambient_dim = point.shape[-1]
        a_param = 1
        b_param = 1 / 2
        quotient = a_param / b_param

        position = point[..., 1:-1, :]
        d_pos = (point[..., 2:, :] - point[..., :-2, :]) / 2
        d_vec = (tangent_vec[..., 2:, :] - tangent_vec[..., :-2, :]) / 2
        d2_pos = point[..., 2:, :] - 2 * point[..., 1:-1, :] + point[..., :-2, :]
        d2_vec = (
            tangent_vec[..., 2:, :]
            - 2 * tangent_vec[..., 1:-1, :]
            + tangent_vec[..., :-2, :]
        )

        vec_a = self.l2_curves_metric.pointwise_norms(
            d_pos, position
        ) ** 2 - 1 / 2 * self.l2_curves_metric.pointwise_inner_products(
            d2_pos, d_pos, position
        )
        vec_b = -2 * self.l2_curves_metric.pointwise_norms(
            d_pos, position
        ) ** 2 - quotient**2 * (
            self.l2_curves_metric.pointwise_norms(d2_pos, position) ** 2
            - self.l2_curves_metric.pointwise_inner_products(d2_pos, d_pos, position)
            ** 2
            / self.l2_curves_metric.pointwise_norms(d_pos, position) ** 2
        )
        vec_c = self.l2_curves_metric.pointwise_norms(
            d_pos, position
        ) ** 2 + 1 / 2 * self.l2_curves_metric.pointwise_inner_products(
            d2_pos, d_pos, position
        )
        vec_d = self.l2_curves_metric.pointwise_norms(d_pos, position) * (
            self.l2_curves_metric.pointwise_inner_products(d2_vec, d_pos, position)
            - (quotient**2 - 1)
            * self.l2_curves_metric.pointwise_inner_products(d_vec, d2_pos, position)
            + (quotient**2 - 2)
            * self.l2_curves_metric.pointwise_inner_products(d2_pos, d_pos, position)
            * self.l2_curves_metric.pointwise_inner_products(d_vec, d_pos, position)
            / self.l2_curves_metric.pointwise_norms(d_pos, position) ** 2
        )

        linear_system = (
            from_vector_to_diagonal_matrix(vec_a[..., :-1], 1)
            + from_vector_to_diagonal_matrix(vec_b, 0)
            + from_vector_to_diagonal_matrix(vec_c[..., 1:], -1)
        )
        vertical_norm = gs.to_ndarray(gs.linalg.solve(linear_system, vec_d), to_ndim=2)
        n_points = vertical_norm.shape[0]
        vertical_norm = gs.squeeze(
            gs.hstack((gs.zeros((n_points, 1)), vertical_norm, gs.zeros((n_points, 1))))
        )

        unit_speed = gs.einsum(
            "...ij,...i->...ij",
            d_pos,
            1 / self.l2_curves_metric.pointwise_norms(d_pos, position),
        )
        tangent_vec_ver = gs.einsum(
            "...ij,...i->...ij", unit_speed, vertical_norm[..., 1:-1]
        )
        tangent_vec_ver = gs.concatenate(
            (
                gs.zeros((n_points, 1, ambient_dim)),
                gs.to_ndarray(tangent_vec_ver, to_ndim=3),
                gs.zeros((n_points, 1, ambient_dim)),
            ),
            axis=1,
        )
        tangent_vec_ver = gs.squeeze(tangent_vec_ver)
        if return_norm:
            return tangent_vec_ver, vertical_norm

        return tangent_vec_ver

    def horizontal_projection(self, tangent_vec, point):
        """Compute horizontal part of tangent vector at base point.

        Parameters
        ----------
        tangent_vec : array-like,
            shape=[..., k_sampling_points, ambient_dim]
            Tangent vector to decompose into horizontal and vertical parts.
        point : array-like,
            shape=[..., k_sampling_points, ambient_dim]
            Discrete curve, base point of tangent_vec in the manifold of curves.

        Returns
        -------
        tangent_vec_hor : array-like,
            shape=[..., k_sampling_points, ambient_dim]
            Horizontal part of tangent_vec.
        """
        tangent_vec_ver = self.vertical_projection(tangent_vec, point)
        return tangent_vec - tangent_vec_ver

    def horizontal_geodesic(self, initial_point, end_point, threshold=1e-3):
        """Compute horizontal geodesic between two curves.

        The horizontal geodesic is computed by an interative procedure where
        the initial curve stays fixed and the sampling points are moved on the
        end curve to obtain its optimal parametrization with respect to the
        initial curve. This optimal matching algorithm sets current_end_curve
        to be the end curve and iterates three steps:
        1) compute the geodesic between the initial curve and current_end_curve
        2) compute the path of reparametrizations that transforms this geodesic
        into a horizontal path of curves
        3) invert this path of reparametrizations to find the horizontal path
        and update current_end_curve to be its end point.
        The algorithm stops when the new current_end_curve is sufficiently
        close to the former current_end_curve.

        Parameters
        ----------
        initial_point : array-like, shape=[k_sampling_points, ambient_dim]
            Initial discrete curve.
        end_point : array-like, shape=[k_sampling_points, ambient_dim]
            End discrete curve.
        threshold: float
            When the difference between the new end curve and the current end
            curve becomes lower than this threshold, the algorithm stops.
            Optional, default: 1e-3.

        Returns
        -------
        horizontal_path : callable
            Time parametrized horizontal geodesic.
        """
        initial_curve, end_curve = initial_point, end_point
        k_sampling_points = initial_curve.shape[0]
        t_space = gs.linspace(0.0, 1.0, k_sampling_points)
        spline_end_curve = CubicSpline(t_space, end_curve, axis=0)

        def construct_reparametrization(vertical_norm, space_deriv_norm):
            r"""Construct path of reparametrizations.

            Construct path of reparametrizations phi(t, u) that transforms
            a path of curves c(t, u) into a horizontal path of curves, i.e.
            :math:`d/dt c(t, phi(t, u))` is a horizontal vector.

            Parameters
            ----------
            vertical_norm: array-like, shape=[n_times, k_sampling_points]
                Pointwise norm of the vertical part of the time derivative of
                the path of curves.
            space_deriv_norm: array-like, shape=[n_times, k_sampling_points]
                Pointwise norm of the space derivative of the path of curves.

            Returns
            -------
            repar: array-like, shape=[n_times, k_sampling_points]
                Path of parametrizations, such that the path of curves
                composed with the path of parametrizations is a horizontal
                path.
            """
            n_times = gs.shape(vertical_norm)[0] + 1
            repar = gs.to_ndarray(gs.linspace(0.0, 1.0, k_sampling_points), 2)
            for i in range(n_times - 1):
                repar_i = [gs.array(0.0)]
                n_times = gs.cast(gs.array(n_times), vertical_norm.dtype)
                for j in range(1, k_sampling_points - 1):
                    d_repar_plus = repar[-1, j + 1] - repar[-1, j]
                    d_repar_minus = repar[-1, j] - repar[-1, j - 1]
                    if vertical_norm[i, j] > 0:
                        repar_space_deriv = k_sampling_points * d_repar_plus
                    else:
                        repar_space_deriv = k_sampling_points * d_repar_minus
                    repar_time_deriv = (
                        repar_space_deriv * vertical_norm[i, j] / space_deriv_norm[i, j]
                    )
                    repar_i.append(repar[-1, j] + repar_time_deriv / n_times)
                repar_i.append(gs.array(1.0))
                repar_i = gs.to_ndarray(gs.stack(repar_i), to_ndim=2)
                repar = gs.concatenate((repar, repar_i), axis=0)

                test_repar = gs.sum(repar[-1, 2:] - repar[-1, 1:-1] < 0)
                if gs.any(test_repar):
                    print(
                        "Warning: phi(t) is non increasing for at least "
                        "one time t. Solution may be inaccurate."
                    )

            return repar

        def invert_reparametrization(repar, path_of_curves, repar_inverse_end, counter):
            r"""Invert path of reparametrizations.

            Given a path of curves c(t, u) and a path of reparametrizations
            phi(t, u), compute:
            :math:`c(t, phi_inv(t, u))` where `phi_inv(t, .) = phi(t, .)^{-1}`
            The computation for the last time t=1 is done differently, using
            the spline function associated to the end curve and the composition
            of the inverse reparametrizations contained in rep_inverse_end:
            :math:`spline_end_curve ° phi_inv(1, .) ° ... ° phi_inv(0, .)`.

            Parameters
            ----------
            repar: array-like, shape=[n_times, k_sampling_points]
                Path of reparametrizations.
            path_of_curves: array-like,
                shape=[n_times, k_sampling_points, ambient_dim]
                Path of curves.
            repar_inverse_end: list
                List of the inverses of the reparametrizations applied to
                the end curve during the optimal matching algorithm.
            counter: int
                Counter associated to the steps of the optimal matching
                algorithm.

            Returns
            -------
            reparametrized_path: array-like,
                shape=[n_times, k_sampling_points, ambient_dim]
                Path of curves composed with the inverse of the path of
                reparametrizations.
            """
            n_times = repar.shape[0]
            initial_curve = path_of_curves[0]
            reparametrized_path = []
            reparametrized_path.append(initial_curve)
            for i in range(1, n_times - 1):
                spline = CubicSpline(t_space, path_of_curves[i], axis=0)
                repar_inverse = CubicSpline(repar[i], t_space)
                curve_repar = gs.from_numpy(spline(repar_inverse(t_space)))
                curve_repar = gs.cast(curve_repar, repar.dtype)
                reparametrized_path.append(curve_repar)

            repar_inverse_end.append(CubicSpline(repar[-1, :], t_space))
            arg = t_space
            for i in range(counter + 1):
                arg = repar_inverse_end[-1 - i](arg)
            end_curve_repar = gs.from_numpy(spline_end_curve(arg))
            end_curve_repar = gs.cast(end_curve_repar, repar.dtype)
            reparametrized_path.append(end_curve_repar)
            return gs.stack(reparametrized_path)

        def horizontal_path(t):
            """Generate parametrized function for horizontal geodesic.

            Parameters
            ----------
            t: array-like, shape=[n_times,]
                Times at which to compute points of the horizontal geodesic.
            """
            n_times = len(t)
            current_end_curve = gs.copy(end_curve)
            repar_inverse_end = []
            gap = 1.0
            counter = 0

            while gap > threshold:
                srv_geod_fun = self.total_space_metric.geodesic(
                    initial_point=initial_curve, end_point=current_end_curve
                )
                geod = srv_geod_fun(t)

                time_deriv = n_times * (geod[1:] - geod[:-1])
                _, vertical_norm = self.vertical_projection(
                    time_deriv, geod[:-1], return_norm=True
                )

                space_deriv = SRVMetric.space_derivative(geod)
                space_deriv_norm = self.ambient_manifold.metric.norm(space_deriv)

                repar = construct_reparametrization(vertical_norm, space_deriv_norm)

                horizontal_path = invert_reparametrization(
                    repar, geod, repar_inverse_end, counter
                )

                new_end_curve = horizontal_path[-1]
                gap = (
                    gs.sum(
                        gs.linalg.norm(new_end_curve - current_end_curve, axis=-1) ** 2
                    )
                ) ** (1 / 2)
                current_end_curve = gs.copy(new_end_curve)

                counter += 1
            return horizontal_path

        return horizontal_path

    def align(self, point, base_point, threshold=1e-3):
        """Find optimal reparametrization of curve with respect to base curve.

        The new parametrization of curve is optimal in the sense that it is the
        member of its fiber closest to the base curve with respect to the SRVMetric.
        It is found as the end point of the horizontal geodesic starting at the base
        curve and ending at the fiber of curve.

        Parameters
        ----------
        point : array-like, shape=[k_sampling_points, ambient_dim]
            Discrete curve.
        base_point : array-like, shape=[k_sampling_points, ambient_dim]
            Discrete curve.
        threshold: float
            Threshold to use in the algorithm to compute the horizontal geodesic.
            Optional, default: 1e-3.

        Returns
        -------
        reparametrized_curve : array-like, shape=[k_sampling_points, ambient_dim]
            Optimal reparametrization of the curve represented by point.
        """
        horizontal_path = self.horizontal_geodesic(base_point, point, threshold)
        return horizontal_path(1.0)


class SRVQuotientMetric(QuotientMetric):
    """SRV quotient metric on the space of unparametrized curves.

    This is the class for the quotient metric induced by the SRV Metric
    on the shape space of unparametrized curves, i.e. the space of parametrized
    curves quotiented by the group of reparametrizations. In the discrete case,
    reparametrization corresponds to resampling.
    """

    def __init__(self, ambient_manifold, k_sampling_points=10):
        dim = ambient_manifold.dim * k_sampling_points
        bundle = SRVShapeBundle(ambient_manifold, dim)
        super(SRVQuotientMetric, self).__init__(
            fiber_bundle=bundle,
            dim=dim,
            shape=(k_sampling_points,) + ambient_manifold.shape,
        )

    def geodesic(self, initial_point, end_point, threshold=1e-3):
        """Geodesic for the quotient SRV Metric.

        The geodesics between unparametrized curves for the quotient metric are
        projections of the horizontal geodesics in the total space of parameterized
        curves. Since in practice shapes can only be encoded by parametrized curves,
        geodesics are given in the total space.
        """
        return self.fiber_bundle.horizontal_geodesic(
            initial_point, end_point, threshold
        )

    def dist(self, point_a, point_b, n_times=20, threshold=1e-3):
        """Quotient SRV distance between unparametrized curves.

        This is the distance induced by the SRV Metric on the space of unparametrized
        curves. To compute this distance, the second curve is aligned to the first
        curve, i.e. is reparametrized in an optimal way with respect to the first curve,
        and the length of the (horizontal) geodesic linking the two is computed for the
        SRV metric.

        Parameters
        ----------
        point_a : array-like, shape=[k_sampling_points, ambient_dim]
            Discrete curve.
        point_b : array-like, shape=[k_sampling_points, ambient_dim]
            Discrete curve.
        n_times: int
            Number of times used to discretize the horizontal geodesic.
            Optional, default: 20.
        threshold: float
            Stop criterion used in the algorithm to compute the horizontal
            geodesic.
            Optional, default: 1e-3.

        Returns
        -------
        quotient_dist : float
            Quotient distance between the two curves represented by point_a and point_b.
        """
        horizontal_path = self.geodesic(
            initial_point=point_a, end_point=point_b, threshold=threshold
        )
        times = gs.linspace(0.0, 1.0, n_times)
        horizontal_geod = horizontal_path(times)
        horizontal_geod_velocity = n_times * (
            horizontal_geod[:-1] - horizontal_geod[1:]
        )
        velocity_norms = self.total_space_metric.norm(
            horizontal_geod_velocity, horizontal_geod[:-1]
        )
        quotient_dist = gs.sum(velocity_norms) / n_times
        return quotient_dist
