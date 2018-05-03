"""
Base class for Riemannian metrics.
"""

import logging
import numpy as np

import geomstats.vectorization as vectorization

EPSILON = 1e-5


class RiemannianMetric(object):
    """
    Base class for Riemannian metrics.
    Note: this class includes sub- and pseudo- Riemannian metrics.
    """

    def __init__(self, dimension, signature=None):
        assert isinstance(dimension, int) and dimension > 0
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

        n_tangent_vecs_a, _ = tangent_vec_a.shape
        n_tangent_vecs_b, _ = tangent_vec_b.shape
        n_inner_prod_mats, _, _ = inner_prod_mat.shape

        bool_all_same_n = (n_tangent_vecs_a
                           == n_tangent_vecs_b
                           == n_inner_prod_mats)
        bool_a = n_tangent_vecs_a == 1
        bool_b = n_tangent_vecs_b == 1
        bool_inner_prod = n_inner_prod_mats == 1
        assert (bool_all_same_n
                or n_tangent_vecs_a == n_tangent_vecs_b and bool_inner_prod
                or n_tangent_vecs_a == n_inner_prod_mats and bool_b
                or n_tangent_vecs_b == n_inner_prod_mats and bool_a
                or bool_a and bool_b
                or bool_a and bool_inner_prod
                or bool_b and bool_inner_prod)

        aux = np.einsum('ij,ijk->ik', tangent_vec_a, inner_prod_mat)
        inner_prod = np.einsum('ik,ik->i', aux, tangent_vec_b)
        inner_prod = vectorization.to_ndarray(inner_prod, to_ndim=2, axis=1)

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

    def exp(self, tangent_vec, base_point=None):
        """
        Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the Riemannian metric.
        """
        raise NotImplementedError(
                'The Riemannian exponential is not implemented.')

    def log(self, point, base_point=None):
        """
        Riemannian logarithm at point base_point
        of tangent vector tangent_vec wrt the Riemannian metric.
        """
        raise NotImplementedError(
                'The Riemannian logarithm is not implemented.')

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

            tangent_vecs = np.einsum('il,k...->i...',
                                     t,
                                     new_initial_tangent_vec)
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
