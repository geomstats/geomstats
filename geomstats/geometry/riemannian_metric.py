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

    Parameters
    ----------
    dim : int
        Dimension of the manifold.
    shape : tuple of int
        Shape of one element of the manifold.
        Optional, default : (dim, ).
    signature : tuple
        Signature of the metric.
        Optional, default: None.
    default_point_type : str, {'vector', 'matrix'}
        Point type.
        Optional, default: 'vector'.
    """

    def __init__(self, dim, shape=None, signature=None, default_point_type=None):
        super(RiemannianMetric, self).__init__(
            dim=dim, shape=shape, default_point_type=default_point_type
        )
        if signature is None:
            signature = (dim, 0)
        self.signature = signature

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
        mat : array-like, shape=[..., dim, dim]
            Inverse of inner-product matrix.
        """
        metric_matrix = self.metric_matrix(base_point)
        cometric_matrix = gs.linalg.inv(metric_matrix)
        return cometric_matrix

    def inner_product_derivative_matrix(self, base_point=None):
        """Compute derivative of the inner prod matrix at base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim]
            Derivative of inverse of inner-product matrix.
        """
        metric_derivative = gs.autodiff.jacobian(self.metric_matrix)
        return metric_derivative(base_point)

    def christoffels(self, base_point):
        r"""Compute Christoffel symbols of the Levi-Civita connection.

        The Koszul formula defining the Levi-Civita connection gives the
        expression of the Christoffel symbols with respect to the metric:
        :math:`\Gamma^k_{ij}(p) = \frac{1}{2} g^{lk}(
            \partial_i g_{jl} + \partial_j g_{li} - \partial_l g_{ij})`,
        where:
        - :math:`p` represents the base point, and
        - :math:`g` represents the Riemannian metric tensor.

        Parameters
        ----------
        base_point: array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        christoffels: array-like, shape=[..., dim, dim, dim]
            Christoffel symbols.
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

        christoffels = 0.5 * (term_1 + term_2 + term_3)
        return christoffels

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
        inner_prod = gs.einsum("...k,...k->...", aux, tangent_vec_b)
        return inner_prod

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
        inner_coproduct = gs.einsum("...i,...i->...", cotangent_vec_a, vector_2)
        return inner_coproduct

    def hamiltonian(self, state):
        r"""Compute the hamiltonian energy associated to the cometric.

        The Hamiltonian at state :math: `(q, p)` is defined by
        .. math:
                H(q, p) = \frac{1}{2} <p, p>_q
        where :math: `<\cdot, \cdot>_q` is the cometric at :math: `q`.

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
        sq_norm = self.inner_product(vector, vector, base_point)
        return sq_norm

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
        norm = gs.sqrt(sq_norm)
        return norm

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

        sq_dist = self.squared_norm(vector=log, base_point=point_a)
        return sq_dist

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
        dist = gs.sqrt(sq_dist)
        return dist

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
        ndim = len(self.shape)

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
        dist = gs.squeeze(dist)
        return dist

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

        pairwise_dist = geometry.symmetric_matrices.SymmetricMatrices.from_vector(
            gs.array(out)
        )
        return pairwise_dist

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
        neighbors : array-like, shape=[..., dim]
            Neighbors.

        Returns
        -------
        closest_neighbor_index : int
            Index of closest neighbor.
        """
        dist = self.dist(point, neighbors)
        closest_neighbor_index = gs.argmin(dist)

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

    def sectional_curvature(self, tangent_vec_a, tangent_vec_b, base_point=None):
        r"""Compute the sectional curvature.

        For two orthonormal tangent vectors :math:`x,y` at a base point,
        the sectional curvature is defined by :math:`<R(x, y)x, y> =
        <R_x(y), y>`. For non-orthonormal vectors vectors, it is
        :math:`<R(x, y)x, y> / \\|x \\wedge y\\|^2`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., n, n]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., n, n]
            Point in the group. Optional, default is the identity

        Returns
        -------
        sectional_curvature : array-like, shape=[...,]
            Sectional curvature at `base_point`.

        See Also
        --------
        https://en.wikipedia.org/wiki/Sectional_curvature
        """
        curvature = self.curvature(
            tangent_vec_a, tangent_vec_b, tangent_vec_a, base_point
        )
        sectional = self.inner_product(curvature, tangent_vec_b, base_point)
        norm_a = self.squared_norm(tangent_vec_a, base_point)
        norm_b = self.squared_norm(tangent_vec_b, base_point)
        inner_ab = self.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        normalization_factor = norm_a * norm_b - inner_ab**2

        condition = gs.isclose(normalization_factor, 0.0)
        normalization_factor = gs.where(condition, EPSILON, normalization_factor)
        return gs.where(~condition, sectional / normalization_factor, 0.0)
