"""
Base class for Riemannian metrics.
"""

import tensorflow as tf
import logging
import numpy as np
import keras.backend as K

import geomstats.vectorization as vectorization

EPSILON = 1e-5


class RiemannianMetric(object):
    """
    Base class for Riemannian metrics.
    Note: this class includes sub- and pseudo- Riemannian metrics.
    """

    def __init__(self, dimension, signature=None):
        assert dimension > 0
        self.dimension = dimension
        if signature is not None:
            assert np.sum(signature) == dimension
        self.signature = signature

    def inner_product_matrix(self, base_point=None):
        """
        Matrix of the inner product defined by the Riemmanian metric
        at point base_point of the manifold.
        """
        raise NotImplementedError(
                'The computation of the inner product matrix'
                ' is not implemented.')

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """
        Inner product defined by the Riemannian metric at point base_point
        between tangent vectors tangent_vec_a and tangent_vec_b.
        """
        tangent_vec_a = vectorization.to_ndarray(tangent_vec_a, to_ndim=2)
        tangent_vec_b = vectorization.to_ndarray(tangent_vec_b, to_ndim=2)

        inner_prod_mat = self.inner_product_matrix(base_point)
        inner_prod_mat = vectorization.to_ndarray(inner_prod_mat, to_ndim=3)
        print(tangent_vec_a)
        n_tangent_vecs_a = tf.shape(tangent_vec_a)[0]

        n_tangent_vecs_b = tf.shape(tangent_vec_b)[0]
        n_inner_prod_mats = tf.shape(inner_prod_mat)[0]

        bool_all_same_n = (n_tangent_vecs_a
                           == n_tangent_vecs_b
                           == n_inner_prod_mats)
        bool_a = n_tangent_vecs_a == 1
        bool_b = n_tangent_vecs_b == 1
        bool_inner_prod = n_inner_prod_mats == 1
        #assert (bool_all_same_n
        #        or n_tangent_vecs_a == n_tangent_vecs_b and bool_inner_prod
        #        or n_tangent_vecs_a == n_inner_prod_mats and bool_b
        #        or n_tangent_vecs_b == n_inner_prod_mats and bool_a
        #        or bool_a and bool_b
        #        or bool_a and bool_inner_prod
        #        or bool_b and bool_inner_prod)

        n_inner_prods = K.max([n_tangent_vecs_a,
                               n_tangent_vecs_b,
                               n_inner_prod_mats],
                               axis=0)
        inner_prod = K.zeros((n_inner_prods, 1))
        for i in range(n_inner_prods):
            tangent_vec_a_i = (tangent_vec_a[0] if n_tangent_vecs_a == 1
                               else tangent_vec_a[i])
            tangent_vec_b_i = (tangent_vec_b[0] if n_tangent_vecs_b == 1
                               else tangent_vec_b[i])
            inner_prod_mat_i = (inner_prod_mat[0] if n_inner_prod_mats == 1
                                else inner_prod_mat[i])
            inner_prod[i] = K.dot(np.dot(tangent_vec_a_i, inner_prod_mat_i),
                                  tangent_vec_b_i.transpose())
        return inner_prod

    def squared_norm(self, vector, base_point=None):
        """
        Squared norm associated to the inner product.

        Note: the squared norm may be non-positive if the metric
        is not positive-definite.
        """
        sq_norm = self.inner_product(vector, vector, base_point)
        return sq_norm

    def norm(self, vector, base_point=None):
        """
        Norm associated to the inner product.
        """
        n_negative_eigenvalues = self.signature[1]
        if n_negative_eigenvalues > 0:
            raise ValueError(
                    'The method \'norm\' only works for positive-definite'
                    ' Riemannian metrics and inner products.')
        sq_norm = self.squared_norm(vector, base_point)
        norm = np.sqrt(sq_norm)
        return norm

    def exp_basis(self, tangent_vec, base_point=None):
        """
        Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the Riemannian metric.
        """
        raise NotImplementedError(
                'The basis function for the Riemannian exponential'
                'is not implemented.')

    def log_basis(self, point, base_point=None):
        """
        Riemannian logarithm at point base_point
        of tangent vector tangent_vec wrt the Riemannian metric.
        """
        raise NotImplementedError(
                'The basis function for the Riemannian logarithm'
                ' is not implemented.')

    def exp(self, tangent_vec, base_point=None):
        """
        Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the Riemannian metric.
        """
        if tangent_vec.ndim == 1:
            tangent_vec = np.expand_dims(tangent_vec, axis=0)
        assert tangent_vec.ndim == 2

        if base_point.ndim == 1:
            base_point = np.expand_dims(base_point, axis=0)
        assert base_point.ndim == 2

        n_tangent_vecs, _ = tangent_vec.shape
        n_base_points, point_dim = base_point.shape

        assert (n_tangent_vecs == n_base_points
                or n_tangent_vecs == 1
                or n_base_points == 1)

        n_exps = np.maximum(n_tangent_vecs, n_base_points)
        exp = np.zeros((n_exps, point_dim))
        for i in range(n_exps):
            base_point_i = (base_point[0] if n_base_points == 1
                            else base_point[i])
            tangent_vec_i = (tangent_vec[0] if n_tangent_vecs == 1
                             else tangent_vec[i])
            exp[i] = self.exp_basis(tangent_vec_i, base_point_i)

        return exp

    def log(self, point, base_point=None):
        """
        Riemannian logarithm at point base_point
        of tangent vector tangent_vec wrt the Riemannian metric.
        """
        point = vectorization.to_ndarray(point, to_ndim=2)
        base_point = vectorization.to_ndarray(base_point, to_ndim=2)

        n_points, _ = point.shape
        n_base_points, point_dim = base_point.shape

        assert (n_points == n_base_points
                or n_points == 1
                or n_base_points == 1)

        n_logs = np.maximum(n_points, n_base_points)
        log = np.zeros((n_logs, point_dim))
        for i in range(n_logs):
            base_point_i = (base_point[0] if n_base_points == 1
                            else base_point[i])
            point_i = (point[0] if n_points == 1
                       else point[i])
            log[i] = self.log_basis(point_i, base_point_i)

        return log

    def geodesic(self, initial_point,
                 end_point=None, initial_tangent_vec=None, point_ndim=1):
        """
        Geodesic curve associated to the Riemannian metric,
        starting at the point initial_point in the direction
        of the initial tangent vector.

        The geodesic is returned as a function of t, which represents the
        geodesic curve parameterized by t.

        By default, the function assumes that points and tangent_vecs are
        represented by vectors: point_ndim=1. This function is overwritten
        for manifolds whose points are represented by matrices or higher
        dimensional tensors.
        """
        initial_point = vectorization.to_ndarray(initial_point,
                                                 to_ndim=point_ndim+1)

        if end_point is None and initial_tangent_vec is None:
            raise ValueError('Specify an end point or an initial tangent '
                             'vector to define the geodesic.')
        if end_point is not None:
            end_point = vectorization.to_ndarray(end_point,
                                                 to_ndim=point_ndim+1)
            shooting_tangent_vec = self.log(point=end_point,
                                            base_point=initial_point)
            if initial_tangent_vec is not None:
                assert np.allclose(shooting_tangent_vec, initial_tangent_vec)
            initial_tangent_vec = shooting_tangent_vec
        initial_tangent_vec = vectorization.to_ndarray(initial_tangent_vec,
                                                       to_ndim=point_ndim+1)

        def point_on_geodesic(t):
            t = vectorization.to_ndarray(t, to_ndim=1)
            t = vectorization.to_ndarray(t, to_ndim=2, axis=1)

            new_initial_point = vectorization.to_ndarray(
                                          initial_point,
                                          to_ndim=point_ndim+1)
            new_initial_tangent_vec = vectorization.to_ndarray(
                                          initial_tangent_vec,
                                          to_ndim=point_ndim+1)

            n_times_t, _ = t.shape
            tangent_vec_shape = new_initial_tangent_vec.shape[1:]
            tangent_vecs = np.zeros((n_times_t,) + tangent_vec_shape)

            for i in range(n_times_t):
                tangent_vecs[i] = t[i] * new_initial_tangent_vec
            point_at_time_t = self.exp(tangent_vec=tangent_vecs,
                                       base_point=new_initial_point)
            return point_at_time_t

        return point_on_geodesic

    def squared_dist(self, point_a, point_b):
        """
        Squared Riemannian distance between points point_a and point_b.

        Note: the squared distance may be non-positive if the metric
        is not positive-definite.
        """
        log = self.log(point=point_b, base_point=point_a)
        sq_dist = self.squared_norm(vector=log, base_point=point_a)
        return sq_dist

    def dist(self, point_a, point_b):
        """
        Riemannian distance between points point_a and point_b. This
        is the geodesic distance associated to the Riemannian metric.
        """
        n_negative_eigenvalues = self.signature[1]
        if n_negative_eigenvalues > 0:
            raise ValueError(
                    'The method \'dist\' only works for positive-definite'
                    ' Riemannian metrics and inner products.')
        sq_dist = self.squared_dist(point_a, point_b)
        dist = np.sqrt(sq_dist)
        return dist

    def variance(self, points, weights=None, base_point=None):
        """
        Weighted variance of the points in the tangent space
        at the base_point.
        """
        n_points = len(points)
        assert n_points > 0

        if weights is None:
            weights = np.ones(n_points)

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
             weights=None, n_max_iterations=100, epsilon=EPSILON):
        """
        Weighted Frechet mean of the points, iterating 3 steps:
        - Project the points on the tangent space with the riemannian log
        - Calculate the tangent mean on the tangent space
        - Shoot the tangent mean onto the manifold with the riemannian exp

        Initialization with one of the points.
        """
        # TODO(nina): profile this code to study performance,
        # i.e. what to do with sq_dists_between_iterates.

        n_points = len(points)
        assert n_points > 0

        if weights is None:
            weights = np.ones(n_points)

        n_weights = len(weights)
        assert n_points == n_weights
        sum_weights = np.sum(weights)

        mean = points[0]
        if n_points == 1:
            return mean

        sq_dists_between_iterates = []
        iteration = 0
        while iteration < n_max_iterations:
            a_tangent_vector = self.log(mean, mean)
            tangent_mean = np.zeros_like(a_tangent_vector)

            for i in range(n_points):
                # TODO(nina): abandon the for loop
                point_i = points[i]
                weight_i = weights[i]
                tangent_mean = tangent_mean + weight_i * self.log(
                                                    point=point_i,
                                                    base_point=mean)
            tangent_mean /= sum_weights

            mean_next = self.exp(tangent_vec=tangent_mean, base_point=mean)

            sq_dist = self.squared_dist(mean_next, mean)
            sq_dists_between_iterates.append(sq_dist)

            variance = self.variance(points=points,
                                     weights=weights,
                                     base_point=mean_next)
            if sq_dist <= epsilon * variance:
                break

            mean = mean_next
            iteration += 1

        if iteration is n_max_iterations:
            logging.warning('Maximum number of iterations {} reached.'
                            'The mean may be inaccurate'
                            ''.format(n_max_iterations))
        return mean

    def tangent_pca(self, points, base_point=None):
        """
        Tangent Principal Component Analysis (tPCA) at base_point.
        This is standard PCA on the Riemannian Logarithms of the points
        at the base point.
        """
        # TODO(nina): It only works for points of ndim=2, adapt to other ndims.
        if base_point is None:
            base_point = self.mean(points)

        tangent_vecs = self.log(points, base_point=base_point)

        covariance_mat = np.cov(tangent_vecs.transpose())
        eigenvalues, tangent_eigenvecs = np.linalg.eig(covariance_mat)

        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        tangent_eigenvecs = tangent_eigenvecs[idx]

        return eigenvalues, tangent_eigenvecs
