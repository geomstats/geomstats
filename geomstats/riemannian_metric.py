"""
Base class for Riemannian metrics.
"""

import logging
import numpy as np

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

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None,):
        """
        Inner product defined by the Riemannian metric at point base_point
        between tangent vectors tangent_vec_a and tangent_vec_b.
        """
        inner_prod_mat = self.inner_product_matrix(base_point)
        inner_prod = np.dot(np.dot(tangent_vec_a.transpose(), inner_prod_mat),
                            tangent_vec_b)
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
                point_i = points[i]
                weight_i = weights[i]

                tangent_mean += weight_i * self.log(point=point_i,
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
