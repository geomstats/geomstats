"""Riemannian and pseudo-Riemannian metrics.

Lead author: Nina Miolane.
"""
from abc import ABC

import joblib

import geomstats.backend as gs
import geomstats.geometry as geometry
from geomstats.geometry.connection import Connection

EPSILON = 1e-4
N_CENTERS = 10
N_REPETITIONS = 20
N_MAX_ITERATIONS = 50000
N_STEPS = 10


class RiemannianMetric(Connection, ABC):
    """Class for Riemannian and pseudo-Riemannian metrics.

    The associated Levi-Civita connection on the tangent bundle.
    """

    def __init__(self, space, signature=None):
        super().__init__(space=space)
        if signature is None:
            signature = (space.dim, 0)
        self.signature = signature

    def __mul__(self, scalar):
        """Multiply the metric by a scalar.

        This method multiplies the (0,2) metric tensor by a scalar. Note that this does
        not scale distances by the scalar. That would require multiplication by the
        square of the scalar.

        Parameters
        ----------
        scalar : float
            The number by which to multiply the metric.

        Returns
        -------
        metric : ScalarProductMetric
            The metric multiplied by the scalar
        """
        from geomstats.geometry.scalar_product_metric import ScalarProductMetric

        if not isinstance(scalar, float):
            return NotImplemented
        return ScalarProductMetric(self, scalar)

    def __rmul__(self, scalar):
        """Multiply the metric by a scalar.

        This method multiplies the (0,2) metric tensor by a scalar. Note that this does
        not scale distances by the scalar. That would require multiplication by the
        square of the scalar.

        Parameters
        ----------
        scalar : float
            The number by which to multiply the metric.

        Returns
        -------
        metric : ScalarProductMetric
            The metric multiplied by the scalar.
        """
        return self * scalar

    def metric_matrix(self, base_point=None):
        """Metric matrix at the tangent space at a base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        raise NotImplementedError(
            "The computation of the metric matrix" " is not implemented."
        )

    def cometric_matrix(self, base_point=None):
        """Inner co-product matrix at the cotangent space at a base point.

        This represents the cometric matrix, i.e. the inverse of the
        metric matrix.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        cometric_matrix : array-like, shape=[..., dim, dim]
            Inverse of inner-product matrix.
        """
        metric_matrix = self.metric_matrix(base_point)
        return gs.linalg.inv(metric_matrix)

    def inner_product_derivative_matrix(self, base_point=None):
        r"""Compute derivative of the inner prod matrix at base point.

        Writing :math:`g_{ij}` the inner-product matrix at base point,
        this computes :math:`mat_{ijk} = \partial_k g_{ij}`, where the
        index k of the derivation is put last.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        metric_derivative : array-like, shape=[..., dim, dim, dim]
            Derivative of the inner-product matrix, where the index
            k of the derivation is last: math:`mat_{ijk} = \partial_k g_{ij}`.
        """
        return gs.autodiff.jacobian_vec(self.metric_matrix)(base_point)

    def christoffels(self, base_point):
        r"""Compute Christoffel symbols of the Levi-Civita connection.

        The Koszul formula defining the Levi-Civita connection gives the
        expression of the Christoffel symbols with respect to the metric:
        :math:`\Gamma^k_{ij}(p) = \frac{1}{2} g^{lk}(
        \partial_i g_{jl} + \partial_j g_{li} - \partial_l g_{ij})`,
        where:

        - :math:`p` represents the base point, and
        - :math:`g` represents the Riemannian metric tensor.

        Note that the function computing the derivative of the metric matrix
        puts the index of the derivation last.

        Parameters
        ----------
        base_point: array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        christoffels: array-like, shape=[..., dim, dim, dim]
            Christoffel symbols, where the contravariant index is first.
        """
        cometric_mat_at_point = self.cometric_matrix(base_point)
        metric_derivative_at_point = self.inner_product_derivative_matrix(base_point)

        term_1 = gs.einsum(
            "...lk,...jli->...kij", cometric_mat_at_point, metric_derivative_at_point
        )
        term_2 = gs.einsum(
            "...lk,...lij->...kij", cometric_mat_at_point, metric_derivative_at_point
        )
        term_3 = -gs.einsum(
            "...lk,...ijl->...kij", cometric_mat_at_point, metric_derivative_at_point
        )

        return 0.5 * (term_1 + term_2 + term_3)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Inner product between two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a: array-like, shape=[..., dim]
            Tangent vector at base point.
        tangent_vec_b: array-like, shape=[..., dim]
            Tangent vector at base point.
        base_point: array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """
        inner_prod_mat = self.metric_matrix(base_point)
        aux = gs.einsum("...j,...jk->...k", tangent_vec_a, inner_prod_mat)
        return gs.dot(aux, tangent_vec_b)

    def inner_coproduct(self, cotangent_vec_a, cotangent_vec_b, base_point):
        """Compute inner coproduct between two cotangent vectors at base point.

        This is the inner product associated to the cometric matrix.

        Parameters
        ----------
        cotangent_vec_a : array-like, shape=[..., dim]
            Cotangent vector at `base_point`.
        cotangent_vet_b : array-like, shape=[..., dim]
            Cotangent vector at `base_point`.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        inner_coproduct : float
            Inner coproduct between the two cotangent vectors.
        """
        vector_2 = gs.einsum(
            "...ij,...j->...i", self.cometric_matrix(base_point), cotangent_vec_b
        )
        return gs.dot(cotangent_vec_a, vector_2)

    def hamiltonian(self, state):
        r"""Compute the hamiltonian energy associated to the cometric.

        The Hamiltonian at state :math:`(q, p)` is defined by

        .. math::
            H(q, p) = \frac{1}{2} <p, p>_q

        where :math:`<\cdot, \cdot>_q` is the cometric at :math:`q`.

        Parameters
        ----------
        state : tuple of arrays
            Position and momentum variables. The position is a point on the
            manifold, while the momentum is cotangent vector.

        Returns
        -------
        energy : float
            Hamiltonian energy at `state`.
        """
        position, momentum = state
        return 1.0 / 2 * self.inner_coproduct(momentum, momentum, position)

    def squared_norm(self, vector, base_point=None):
        """Compute the square of the norm of a vector.

        Squared norm of a vector associated to the inner product
        at the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        sq_norm : array-like, shape=[...,]
            Squared norm.
        """
        return self.inner_product(vector, vector, base_point)

    def norm(self, vector, base_point=None):
        """Compute norm of a vector.

        Norm of a vector associated to the inner product
        at the tangent space at a base point.

        Note: This only works for positive-definite
        Riemannian metrics and inner products.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        norm : array-like, shape=[...,]
            Norm.
        """
        sq_norm = self.squared_norm(vector, base_point)
        return gs.sqrt(sq_norm)

    def normalize(self, vector, base_point):
        """Normalize tangent vector at a given point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Tangent vector at base_point.
        base_point : array-like, shape=[..., dim]
            Point.

        Returns
        -------
        normalized_vector : array-like, shape=[..., dim]
            Unit tangent vector at base_point.
        """
        norm = self.norm(vector, base_point)
        norm = gs.where(norm == 0, gs.ones(norm.shape), norm)
        return gs.einsum("...i,...->...i", vector, 1 / norm)

    def random_unit_tangent_vec(self, base_point, n_vectors=1):
        """Generate a random unit tangent vector at a given point.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point.
        n_vectors : float
            Number of vectors to be generated at base_point.
            For vectorization purposes n_vectors can be greater than 1 iff
            base_point consists of a single point.

        Returns
        -------
        normalized_vector : array-like, shape=[..., n_vectors, dim]
            Random unit tangent vector at base_point.
        """
        shape = base_point.shape
        if len(shape) > len(self.shape) and n_vectors > 1:
            raise ValueError(
                "Several tangent vectors is only applicable to a single base point."
            )
        vec_shape = (n_vectors, *shape) if n_vectors > 1 else shape
        random_vector = gs.random.rand(*vec_shape)
        return self.normalize(random_vector, base_point)

    def squared_dist(self, point_a, point_b, **kwargs):
        """Squared geodesic distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[..., dim]
            Point.
        point_b : array-like, shape=[..., dim]
            Point.

        Returns
        -------
        sq_dist : array-like, shape=[...,]
            Squared distance.
        """
        log = self.log(point=point_b, base_point=point_a, **kwargs)

        return self.squared_norm(vector=log, base_point=point_a)

    def dist(self, point_a, point_b, **kwargs):
        """Geodesic distance between two points.

        Note: It only works for positive definite
        Riemannian metrics.

        Parameters
        ----------
        point_a : array-like, shape=[..., dim]
            Point.
        point_b : array-like, shape=[..., dim]
            Point.

        Returns
        -------
        dist : array-like, shape=[...,]
            Distance.
        """
        sq_dist = self.squared_dist(point_a, point_b, **kwargs)
        return gs.sqrt(sq_dist)

    def dist_broadcast(self, point_a, point_b):
        """Compute the geodesic distance between points.

        If n_samples_a == n_samples_b then dist is the element-wise
        distance result of a point in points_a with the point from
        points_b of the same index. If n_samples_a not equal to
        n_samples_b then dist is the result of applying geodesic
        distance for each point from points_a to all points from
        points_b.

        Parameters
        ----------
        point_a : array-like, shape=[n_samples_a, dim]
            Set of points in the Poincare ball.
        point_b : array-like, shape=[n_samples_b, dim]
            Second set of points in the Poincare ball.

        Returns
        -------
        dist : array-like,
            shape=[n_samples_a, dim] or [n_samples_a, n_samples_b, dim]
            Geodesic distance between the two points.
        """
        ndim = len(self._space.shape)

        if point_a.shape[-ndim:] != point_b.shape[-ndim:]:
            raise ValueError("Manifold dimensions not equal")

        if ndim in (point_a.ndim, point_b.ndim) or (point_a.shape == point_b.shape):
            return self.dist(point_a, point_b)

        n_samples = point_a.shape[0] * point_b.shape[0]
        point_a_broadcast, point_b_broadcast = gs.broadcast_arrays(
            point_a[:, None], point_b[None, ...]
        )

        point_a_flatten = gs.reshape(
            point_a_broadcast, (n_samples,) + point_a.shape[-ndim:]
        )
        point_b_flatten = gs.reshape(
            point_b_broadcast, (n_samples,) + point_a.shape[-ndim:]
        )

        dist = self.dist(point_a_flatten, point_b_flatten)
        dist = gs.reshape(dist, (point_a.shape[0], point_b.shape[0]))
        return gs.squeeze(dist)

    def dist_pairwise(self, points, n_jobs=1, **joblib_kwargs):
        """Compute the pairwise distance between points.

        Parameters
        ----------
        points : array-like, shape=[n_samples, dim]
            Set of points in the manifold.
        n_jobs : int
            Number of jobs to run in parallel, using joblib. Note that a
            higher number of jobs may not be beneficial when one computation
            of a geodesic distance is cheap.
            Optional. Default: 1.
        **joblib_kwargs : dict
            Keyword arguments to joblib.Parallel

        Returns
        -------
        dist : array-like, shape=[n_samples, n_samples]
            Pairwise distance matrix between all the points.

        See Also
        --------
        `joblib documentations <https://joblib.readthedocs.io/en/latest/>`_
        """
        n_samples = points.shape[0]
        rows, cols = gs.triu_indices(n_samples)

        @joblib.delayed
        @joblib.wrap_non_picklable_objects
        def pickable_dist(x, y):
            """Wrap distance function to make it pickable."""
            return self.dist(x, y)

        pool = joblib.Parallel(n_jobs=n_jobs, **joblib_kwargs)
        out = pool(pickable_dist(points[i], points[j]) for i, j in zip(rows, cols))

        return geometry.symmetric_matrices.SymmetricMatrices.from_vector(gs.array(out))

    def diameter(self, points):
        """Give the distance between two farthest points.

        Distance between the two points that are farthest away from each other
        in points.

        Parameters
        ----------
        points : array-like, shape=[..., dim]
            Points.

        Returns
        -------
        diameter : float
            Distance between two farthest points.
        """
        diameter = 0.0
        n_points = points.shape[0]

        for i in range(n_points - 1):
            dist_to_neighbors = self.dist(points[i, :], points[i + 1 :, :])
            dist_to_farthest_neighbor = gs.amax(dist_to_neighbors)
            diameter = gs.maximum(diameter, dist_to_farthest_neighbor)

        return diameter

    def closest_neighbor_index(self, point, neighbors):
        """Closest neighbor of point among neighbors.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point.
        neighbors : array-like, shape=[n_neighbors, dim]
            Neighbors.

        Returns
        -------
        closest_neighbor_index : int
            Index of closest neighbor.
        """
        n_points = point.shape[0] if gs.ndim(point) == gs.ndim(neighbors) else 1
        n_neighbors = neighbors.shape[0]

        if n_points > 1 and n_neighbors > 1:
            neighbors = gs.repeat(neighbors, n_points, axis=0)

            point = gs.concatenate([point for _ in range(n_neighbors)])

        closest_neighbor_index = gs.argmin(
            gs.transpose(
                gs.reshape(self.dist(point, neighbors), (n_neighbors, n_points)),
            ),
            axis=1,
        )

        if n_points == 1:
            return closest_neighbor_index[0]

        return closest_neighbor_index

    def normal_basis(self, basis, base_point=None):
        """Normalize the basis with respect to the metric.

        This corresponds to a renormalization of each basis vector.

        Parameters
        ----------
        basis : array-like, shape=[dim, dim]
            Matrix of a metric.
        base_point

        Returns
        -------
        basis : array-like, shape=[dim, n, n]
            Normal basis.
        """
        norms = self.squared_norm(basis, base_point)

        return gs.einsum("i, ikl->ikl", 1.0 / gs.sqrt(norms), basis)

    def covariant_riemann_tensor(self, base_point):
        r"""Compute purely covariant version of Riemannian tensor at base_point.

        In the literature the covariant riemannian tensor is noted :math:`R_{ijkl}`.

        Convention used in the literature (tensor index notation, ref. Wikipedia) is:
        :math:`R_{ijkl} = <R(x_k, x_l)x_j, x_i>`
        where :math:`x_i` is the i-th basis vector of the tangent space at base point.

        In other words:
        [cov_riemann_tensor]_{ijkl} = [metric_matrix]_{im} [riemann_tensor]_{jkl}^m

        Parameters
        ----------
        base_point :  array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        covariant_tensor : array-like, shape=[..., dim, dim, dim, dim]
            covariant_riemann_tensor[..., i, j, k, l] = R_{ijkl}
            Covariant version of Riemannian curvature tensor.
        """
        riemann_tensor = self.riemann_tensor(base_point)
        metric = self.metric_matrix(base_point)
        return gs.einsum("...ij, ...klmj->...iklm", metric, riemann_tensor)

    def sectional_curvature(self, tangent_vec_a, tangent_vec_b, base_point=None):
        r"""Compute the sectional curvature.

        In the literature sectional curvature is noted K.

        For two orthonormal tangent vectors :math:`x,y` at a base point,
        the sectional curvature is defined by :math:`K(x,y) = <R(x, y)x, y>`.

        For non-orthonormal vectors, it is
        :math:`K(x,y) = <R(x, y)y, x> / (<x, x><y, y> - <x, y>^2)`.

        sectional_curvature(X, Y, P) = K(X,Y) where X, Y are tangent vectors
        at base point P.

        The information manifold of multinomial distributions has constant
        sectional curvature given by :math:`K = 2 \sqrt{n}`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., dim]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., dim]
            Point in the manifold.

        Returns
        -------
        sectional_curvature : array-like, shape=[...,]
            Sectional curvature at `base_point`.
        """
        curvature = self.curvature(
            tangent_vec_a, tangent_vec_b, tangent_vec_b, base_point
        )
        sectional = self.inner_product(curvature, tangent_vec_a, base_point)
        norm_a = self.squared_norm(tangent_vec_a, base_point)
        norm_b = self.squared_norm(tangent_vec_b, base_point)
        inner_ab = self.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        normalization_factor = norm_a * norm_b - inner_ab**2

        condition = gs.isclose(normalization_factor, 0.0)
        normalization_factor = gs.where(condition, EPSILON, normalization_factor)
        return gs.where(~condition, sectional / normalization_factor, 0.0)

    def scalar_curvature(self, base_point):
        r"""Compute scalar curvature at base_point.

        In the literature scalar_curvature is noted S and writes:
        :math:`S = g^{ij} Ric_{ij}`,
        with Einstein notation, where we recognize the trace of the Ricci
        tensor according to the Riemannian metric :math:`g`.

        Parameters
        ----------
        base_point :  array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        curvature : array-like, shape=[...,]
            Scalar curvature.
        """
        ricci_tensor = self.ricci_tensor(base_point)
        cometric_matrix = self.cometric_matrix(base_point)
        return gs.einsum("...ij, ...ij -> ...", cometric_matrix, ricci_tensor)
