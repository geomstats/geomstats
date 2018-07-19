"""
Riemannian and pseudo-Riemannian metrics.
"""

import geomstats.backend as gs


EPSILON = 1e-4


def loss(y_pred, y_true, metric):
    """
    Loss function given by a riemannian metric,
    expressed as the squared geodesic distance between the prediction
    and the ground truth.
    """
    loss = metric.squared_dist(y_pred, y_true)
    return loss


def grad(y_pred, y_true, metric):
    """
    Closed-form for the gradient of the loss function.
    """
    tangent_vec = metric.log(base_point=y_pred, point=y_true)
    grad_vec = - 2. * tangent_vec

    inner_prod_mat = metric.inner_product_matrix(base_point=y_pred)

    grad = gs.dot(grad_vec, gs.transpose(inner_prod_mat, axes=(0, 2, 1)))
    return grad


class RiemannianMetric(object):
    """
    Class for Riemannian and pseudo-Riemannian metrics.
    """
    def __init__(self, dimension, signature=None):
        assert isinstance(dimension, int) and dimension > 0
        self.dimension = dimension
        self.signature = signature

    def inner_product_matrix(self, base_point=None):
        """
        Inner product matrix at the tangent space at a base point.
        """
        raise NotImplementedError(
                'The computation of the inner product matrix'
                ' is not implemented.')

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """
        Inner product between two tangent vectors at a base point.
        """
        tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=2)
        tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=2)
        n_tangent_vec_a, _ = tangent_vec_a.shape
        n_tangent_vec_b, _ = tangent_vec_b.shape

        inner_prod_mat = self.inner_product_matrix(base_point)
        inner_prod_mat = gs.to_ndarray(inner_prod_mat, to_ndim=3)
        n_mats, _, _ = inner_prod_mat.shape

        n_inner_prod = gs.maximum(n_tangent_vec_a, n_tangent_vec_b)
        n_inner_prod = gs.maximum(n_inner_prod, n_mats)

        n_tiles_a = gs.divide(n_inner_prod, n_tangent_vec_a)
        n_tiles_a = gs.cast(n_tiles_a, gs.int32)
        tangent_vec_a = gs.tile(tangent_vec_a, [n_tiles_a, 1])
        n_tiles_b = gs.divide(n_inner_prod, n_tangent_vec_b)
        n_tiles_b = gs.cast(n_tiles_b, gs.int32)
        tangent_vec_b = gs.tile(tangent_vec_b, [n_tiles_b, 1])

        n_tiles_mat = gs.divide(n_inner_prod, n_mats)
        n_tiles_mat = gs.cast(n_tiles_mat, gs.int32)
        inner_prod_mat = gs.tile(inner_prod_mat, [n_tiles_mat, 1, 1])

        aux = gs.einsum('nj,njk->nk', tangent_vec_a, inner_prod_mat)
        inner_prod = gs.einsum('nk,nk->n', aux, tangent_vec_b)
        inner_prod = gs.to_ndarray(inner_prod, to_ndim=2, axis=1)

        assert gs.ndim(inner_prod) == 2, inner_prod.shape
        return inner_prod

    def squared_norm(self, vector, base_point=None):
        """
        Squared norm of a vector associated to the inner product
        at the tangent space at a base point.
        """
        sq_norm = self.inner_product(vector, vector, base_point)
        return sq_norm

    def norm(self, vector, base_point=None):
        """
        Norm of a vector associated to the inner product
        at the tangent space at a base point.
        """
        n_negative_eigenvalues = self.signature[1]
        if n_negative_eigenvalues > 0:
            raise ValueError(
                    'The method \'norm\' only works for positive-definite'
                    ' Riemannian metrics and inner products.')
        sq_norm = self.squared_norm(vector, base_point)
        norm = gs.sqrt(sq_norm)
        return norm

    def exp(self, tangent_vec, base_point=None):
        """
        Riemannian exponential of a tangent vector wrt to a base point.
        """
        raise NotImplementedError(
                'The Riemannian exponential is not implemented.')

    def log(self, point, base_point=None):
        """
        Riemannian logarithm of a point wrt a base point.
        """
        raise NotImplementedError(
                'The Riemannian logarithm is not implemented.')

    def geodesic(self, initial_point,
                 end_point=None, initial_tangent_vec=None, point_ndim=1):
        """
        Geodesic curve defined by either:
        - an initial point and an initial tangent vector,
        or
        -an initial point and an end point.

        The geodesic is returned as a function parameterized by t.
        """
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
            t = gs.to_ndarray(t, to_ndim=1)
            t = gs.to_ndarray(t, to_ndim=2, axis=1)
            new_initial_point = gs.to_ndarray(
                                          initial_point,
                                          to_ndim=point_ndim+1)
            new_initial_tangent_vec = gs.to_ndarray(
                                          initial_tangent_vec,
                                          to_ndim=point_ndim+1)

            tangent_vecs = gs.einsum('il,k...->i...',
                                     t,
                                     new_initial_tangent_vec)
            point_at_time_t = self.exp(tangent_vec=tangent_vecs,
                                       base_point=new_initial_point)
            return point_at_time_t

        return point_on_geodesic

    def squared_dist(self, point_a, point_b):
        """
        Squared geodesic distance between two points.
        """
        log = self.log(point=point_b, base_point=point_a)
        sq_dist = self.squared_norm(vector=log, base_point=point_a)
        return sq_dist

    def dist(self, point_a, point_b):
        """
        Geodesic distance between two points.
        """
        n_negative_eigenvalues = self.signature[1]
        if n_negative_eigenvalues > 0:
            raise ValueError(
                    'The method \'dist\' only works for positive-definite'
                    ' Riemannian metrics and inner products.')
        sq_dist = self.squared_dist(point_a, point_b)
        dist = gs.sqrt(sq_dist)
        return dist

    def variance(self, points, weights=None, base_point=None):
        """
        Variance of (weighted) points wrt a base point.
        """
        n_points = len(points)
        assert n_points > 0

        if weights is None:
            weights = gs.ones(n_points)

        n_weights = len(weights)
        assert n_points == n_weights
        sum_weights = sum(weights)

        if base_point is None:
            base_point = self.mean(points, weights)

        variance = 0

        for i in range(n_points):
            weight_i = weights[i]
            point_i = points[i]

            sq_dist = self.squared_dist(base_point, point_i)

            variance += weight_i * sq_dist
        variance /= sum_weights

        return variance

    def mean(self, points,
             weights=None, n_max_iterations=32, epsilon=EPSILON):
        """
        Frechet mean of (weighted) points.
        """
        # TODO(nina): profile this code to study performance,
        # i.e. what to do with sq_dists_between_iterates.

        n_points = len(points)
        assert n_points > 0

        if weights is None:
            weights = gs.ones(n_points)

        n_weights = len(weights)
        assert n_points == n_weights
        sum_weights = gs.sum(weights)

        mean = points[0]
        if n_points == 1:
            return mean

        sq_dists_between_iterates = []
        iteration = 0
        while iteration < n_max_iterations:
            a_tangent_vector = self.log(mean, mean)
            tangent_mean = gs.zeros_like(a_tangent_vector)

            for i in range(n_points):
                # TODO(nina): abandon the for loop
                point_i = points[i]
                weight_i = weights[i]
                tangent_mean = tangent_mean + weight_i * self.log(
                                                    point=point_i,
                                                    base_point=mean)
            tangent_mean /= sum_weights

            mean_next = self.exp(
                tangent_vec=tangent_mean,
                base_point=mean)

            sq_dist = self.squared_dist(mean_next, mean)
            sq_dists_between_iterates.append(sq_dist)

            variance = self.variance(points=points,
                                     weights=weights,
                                     base_point=mean_next)
            if gs.isclose(variance, 0):
                break
            if sq_dist <= epsilon * variance:
                break

            mean = mean_next
            iteration += 1

        if iteration is n_max_iterations:
            print('Maximum number of iterations {} reached.'
                  'The mean may be inaccurate'.format(n_max_iterations))
        return mean

    def tangent_pca(self, points, base_point=None):
        """
        Tangent Principal Component Analysis (tPCA) of points
        on the tangent space at a base point.
        """
        # TODO(nina): It only works for points of ndim=2, adapt to other ndims.
        if base_point is None:
            base_point = self.mean(points)

        tangent_vecs = self.log(points, base_point=base_point)

        covariance_mat = gs.cov(tangent_vecs.transpose())
        eigenvalues, tangent_eigenvecs = gs.linalg.eig(covariance_mat)

        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        tangent_eigenvecs = tangent_eigenvecs[idx]

        return eigenvalues, tangent_eigenvecs
