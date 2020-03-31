"""Riemannian and pseudo-Riemannian metrics."""

import math

import autograd

import geomstats.backend as gs
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
    y_pred
    y_true
    metric

    Returns
    -------
    loss

    """
    loss = metric.squared_dist(y_pred, y_true)
    return loss


def grad(y_pred, y_true, metric):
    """Closed-form for the gradient of the loss function."""
    tangent_vec = metric.log(base_point=y_pred, point=y_true)
    grad_vec = - 2. * tangent_vec

    inner_prod_mat = metric.inner_product_matrix(base_point=y_pred)

    grad = gs.einsum('ni,nij->ni',
                     grad_vec,
                     gs.transpose(inner_prod_mat, axes=(0, 2, 1)))

    return grad


class RiemannianMetric(Connection):
    """Class for Riemannian and pseudo-Riemannian metrics."""

    def __init__(self, dimension, signature=None):
        assert isinstance(dimension, int) or dimension == math.inf
        assert dimension > 0
        super().__init__(dimension=dimension)
        self.signature = signature

    def inner_product_matrix(self, base_point=None):
        """Inner product matrix at the tangent space at a base point.

        Parameters
        ----------
        base_point : array-like, shape=[n_samples, dimension], optional
        """
        raise NotImplementedError(
            'The computation of the inner product matrix'
            ' is not implemented.')

    def inner_product_inverse_matrix(self, base_point=None):
        """Inner product matrix at the tangent space at a base point.

        Parameters
        ----------
        base_point : array-like, shape=[n_samples, dimension], optional
        """
        metric_matrix = self.inner_product_matrix(base_point)
        cometric_matrix = gs.linalg.inv(metric_matrix)
        return cometric_matrix

    def inner_product_derivative_matrix(self, base_point=None):
        """Compute derivative of the inner prod matrix at base point.

        Parameters
        ----------
        base_point : array-like, shape=[n_samples, dimension], optional
        """
        metric_derivative = autograd.jacobian(self.inner_product_matrix)
        return metric_derivative(base_point)

    def christoffels(self, base_point):
        """Compute Christoffel symbols associated with the connection.

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dimension]

        Returns
        -------
        christoffels: array-like,
                             shape=[n_samples, dimension, dimension, dimension]
        """
        cometric_mat_at_point = self.inner_product_inverse_matrix(base_point)
        metric_derivative_at_point = self.inner_product_derivative_matrix(
            base_point)
        term_1 = gs.einsum('nim,nmkl->nikl',
                           cometric_mat_at_point,
                           metric_derivative_at_point)
        term_2 = gs.einsum('nim,nmlk->nilk',
                           cometric_mat_at_point,
                           metric_derivative_at_point)
        term_3 = - gs.einsum('nim,nklm->nikl',
                             cometric_mat_at_point,
                             metric_derivative_at_point)

        christoffels = 0.5 * (term_1 + term_2 + term_3)
        return christoffels

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Inner product between two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a: array-like, shape=[n_samples, dimension]
                                   or shape=[1, dimension]
        tangent_vec_b: array-like, shape=[n_samples, dimension]
                                   or shape=[1, dimension]
        base_point: array-like, shape=[n_samples, dimension]
                                or shape=[1, dimension]

        Returns
        -------
        inner_product : array-like, shape=[n_samples,]
        """
        tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=2)
        tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=2)
        n_tangent_vec_a = gs.shape(tangent_vec_a)[0]
        n_tangent_vec_b = gs.shape(tangent_vec_b)[0]

        inner_prod_mat = self.inner_product_matrix(base_point)
        inner_prod_mat = gs.to_ndarray(inner_prod_mat, to_ndim=3)
        n_mats = gs.shape(inner_prod_mat)[0]

        if n_tangent_vec_a != n_mats:
            if n_tangent_vec_a == 1:
                tangent_vec_a = gs.squeeze(tangent_vec_a, axis=0)
                einsum_str_a = 'j,njk->nk'
            elif n_mats == 1:
                inner_prod_mat = gs.squeeze(inner_prod_mat, axis=0)
                einsum_str_a = 'nj,jk->nk'
            else:
                raise ValueError('Shape mismatch for einsum.')
        else:
            einsum_str_a = 'nj,njk->nk'

        aux = gs.einsum(einsum_str_a, tangent_vec_a, inner_prod_mat)
        n_auxs, _ = gs.shape(aux)

        if n_tangent_vec_b != n_auxs:
            if n_auxs == 1:
                aux = gs.squeeze(aux, axis=0)
                einsum_str_b = 'k,nk->n'
            elif n_tangent_vec_b == 1:
                tangent_vec_b = gs.squeeze(tangent_vec_b, axis=0)
                einsum_str_b = 'nk,k->n'
            else:
                raise ValueError('Shape mismatch for einsum.')
        else:
            einsum_str_b = 'nk,nk->n'

        inner_prod = gs.einsum(einsum_str_b, aux, tangent_vec_b)
        inner_prod = gs.to_ndarray(inner_prod, to_ndim=2, axis=1)

        assert gs.ndim(inner_prod) == 2, inner_prod.shape
        return inner_prod

    def squared_norm(self, vector, base_point=None):
        """Compute the square of the norm of a vector.

        Squared norm of a vector associated to the inner product
        at the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[n_samples, dimension]
        base_point : array-like, shape=[n_samples, dimension]

        Returns
        -------
        sq_norm : array-like, shape=[n_samples,]
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
        vector : array-like, shape=[n_samples, dimension]
        base_point : array-like, shape=[n_samples, dimension]

        Returns
        -------
        norm : array-like, shape=[n_samples,]
        """
        sq_norm = self.squared_norm(vector, base_point)
        norm = gs.sqrt(sq_norm)
        return norm

    def squared_dist(self, point_a, point_b):
        """Squared geodesic distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[n_samples, dimension]
        point_b : array-like, shape=[n_samples, dimension]

        Returns
        -------
        sq_dist : array-like, shape=[n_samples,]
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
        point_a : array-like, shape=[n_samples, dimension]
        point_b : array-like, shape=[n_samples, dimension]

        Returns
        -------
        dist : array-like, shape=[n_samples,]
        """
        sq_dist = self.squared_dist(point_a, point_b)
        dist = gs.sqrt(sq_dist)
        return dist

    def diameter(self, points):
        """Give the distance between two farthest points.

        Distance between the two points that are farthest away from each other
        in points.

        Parameters
        ----------
        points

        Returns
        -------
        diameter

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
        point
        neighbors
        Returns
        -------
        closest_neighbor_index

        """
        dist = self.dist(point, neighbors)
        closest_neighbor_index = gs.argmin(dist)

        return closest_neighbor_index
