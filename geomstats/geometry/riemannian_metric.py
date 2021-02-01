"""Riemannian and pseudo-Riemannian metrics."""

import autograd
import joblib

import geomstats.backend as gs
import geomstats.vectorization
from geomstats.geometry.connection import Connection

EPSILON = 1e-4
N_CENTERS = 10
TOLERANCE = 1e-5
N_REPETITIONS = 20
N_MAX_ITERATIONS = 50000
N_STEPS = 10


def loss(y_pred, y_true, metric):
    """Compute loss function between prediction and ground truth.

    Loss function given by a Riemannian metric,
    expressed as the squared geodesic distance between the prediction
    and the ground truth.

    Parameters
    ----------
    y_pred : array-like, shape=[..., dim]
        Prediction.
    y_true : array-like, shape=[..., dim]
        Ground-truth.
    metric : RiemannianMetric
        Metric.

    Returns
    -------
    sq_dist : array-like, shape=[...,]
        Loss, i.e. the squared distance.
    """
    sq_dist = metric.squared_dist(y_pred, y_true)
    return sq_dist


def grad(y_pred, y_true, metric):
    """Closed-form for the gradient of the loss function.

    Parameters
    ----------
    y_pred : array-like, shape=[..., dim]
        Prediction.
    y_true : array-like, shape=[..., dim]
        Ground-truth.
    metric : RiemannianMetric
        Metric.

    Returns
    -------
    loss_grad : array-like, shape=[...,]
        Gradient of the loss.

    """
    tangent_vec = metric.log(base_point=y_pred, point=y_true)
    grad_vec = - 2. * tangent_vec

    inner_prod_mat = metric.metric_matrix(base_point=y_pred)
    is_vectorized = inner_prod_mat.ndim == 3
    axes = (0, 2, 1) if is_vectorized else (1, 0)

    loss_grad = gs.einsum(
        '...i,...ij->...i',
        grad_vec,
        gs.transpose(inner_prod_mat, axes=axes))

    return loss_grad


class RiemannianMetric(Connection):
    """Class for Riemannian and pseudo-Riemannian metrics.

    Parameters
    ----------
    dim : int
        Dimension of the manifold.
    signature : tuple
        Signature of the metric.
        Optional, default: None.
    default_point_type : str, {'vector', 'matrix'}
        Point type.
        Optional, default: 'vector'.
    """

    def __init__(self, dim, signature=None, default_point_type='vector'):
        super(RiemannianMetric, self).__init__(
            dim=dim, default_point_type=default_point_type)
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
            'The computation of the metric matrix'
            ' is not implemented.')

    def inner_product_inverse_matrix(self, base_point=None):
        """Inner product matrix at the tangent space at a base point.

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
        metric_derivative = autograd.jacobian(self.metric_matrix)
        return metric_derivative(base_point)

    def christoffels(self, base_point):
        """Compute Christoffel symbols associated with the connection.

        Parameters
        ----------
        base_point: array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        christoffels: array-like, shape=[..., dim, dim, dim]
            Christoffel symbols.
        """
        cometric_mat_at_point = self.inner_product_inverse_matrix(base_point)
        metric_derivative_at_point = self.inner_product_derivative_matrix(
            base_point)
        term_1 = gs.einsum('...im,...mkl->...ikl',
                           cometric_mat_at_point,
                           metric_derivative_at_point)
        term_2 = gs.einsum('...im,...mlk->...ilk',
                           cometric_mat_at_point,
                           metric_derivative_at_point)
        term_3 = - gs.einsum('...im,...klm->...ikl',
                             cometric_mat_at_point,
                             metric_derivative_at_point)

        christoffels = 0.5 * (term_1 + term_2 + term_3)
        return christoffels

    @geomstats.vectorization.decorator(['else', 'vector', 'vector', 'vector'])
    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
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
        inner_prod_mat = gs.to_ndarray(inner_prod_mat, to_ndim=3)

        aux = gs.einsum('...j,...jk->...k', tangent_vec_a, inner_prod_mat)

        inner_prod = gs.einsum('...k,...k->...', aux, tangent_vec_b)
        inner_prod = gs.to_ndarray(inner_prod, to_ndim=1)
        inner_prod = gs.to_ndarray(inner_prod, to_ndim=2, axis=1)

        return inner_prod

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

    def squared_dist(self, point_a, point_b):
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
        log = self.log(point=point_b, base_point=point_a)

        sq_dist = self.squared_norm(vector=log, base_point=point_a)
        return sq_dist

    def dist(self, point_a, point_b):
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
        sq_dist = self.squared_dist(point_a, point_b)
        dist = gs.sqrt(sq_dist)
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
        https://joblib.readthedocs.io/en/latest/
        """
        n_samples = points.shape[0]
        rows, cols = gs.triu_indices(n_samples)

        @joblib.delayed
        @joblib.wrap_non_picklable_objects
        def pickable_dist(x, y):
            """Wrap distance function to make it pickable."""
            return self.dist(x, y)

        pool = joblib.Parallel(n_jobs=n_jobs, **joblib_kwargs)
        out = pool(
            pickable_dist(points[i], points[j]) for i, j in zip(rows, cols))

        pairwise_dist = geomstats.geometry.symmetric_matrices.\
            SymmetricMatrices.from_vector(gs.array(out))

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
            dist_to_neighbors = self.dist(points[i, :], points[i + 1:, :])
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

    def orthonormal_basis(self, basis, base_point=None):
        """Orthonormalize the basis with respect to the metric.

        This corresponds to a renormalization.

        Parameters
        ----------
        basis : array-like, shape=[dim, dim]
            Matrix of a metric.
        base_point

        Returns
        -------
        basis : array-like, shape=[dim, n, n]
            Orthonormal basis.
        """
        norms = self.squared_norm(basis, base_point)

        return gs.einsum('i, ikl->ikl', 1. / gs.sqrt(norms), basis)
