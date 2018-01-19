"""
Base classes for all abstract manifolds.
The OOP structure is inspired by Andrea Censi
in his module geometry.
"""

import numpy as np
import math

EPSILON = 1e-5


class Manifold(object):
    """Base class for differentiable manifolds."""

    def __init__(self, dimension):
        self.dimension = dimension

    def belongs(self, point):
        """Check if the point belongs to the manifold."""
        raise NotImplementedError('belongs is not implemented.')

    def regularize(self, point):
        """
        Regularizes the point's coordinates to the canonical representation
        for this manifold.
        """
        return point


class RiemannianMetric(object):
    """
    Base class for (pseudo-/sub-) Riemannian metrics.
    """
    def riemannian_inner_product_matrix(self, ref_point):
        """
        Compute the matrix of the Riemmanian metric at point ref_point.
        """
        raise NotImplementedError(
                'The computation of the metric matrix is not implemented.')

    def riemannian_inner_product(self, ref_point,
                                 tangent_vec_a, tangent_vec_b):
        """
        Compute the inner product at point ref_point
        between tangent vectors tangent_vec_a and tangent_vec_b.
        """
        inner_prod_mat = self.riemannian_inner_product_matrix(ref_point)
        inner_prod = np.dot(np.dot(tangent_vec_a.transpose(), inner_prod_mat),
                            tangent_vec_b)
        return inner_prod

    def riemannian_squared_norm(self, ref_point, vector):
        """
        Squared norm associated to the inner product.

        Note: squared norm may be non-positive if the metric signature
        is not (0, dimension).
        """
        sq_norm = self.riemannian_inner_product(ref_point, vector, vector)
        return sq_norm

    def riemannian_norm(self, ref_point, vector):
        """
        Norm associated to the inner product.

        Note: Only for Riemannian metrics, i.e. signature (0, dimension).
        """
        sq_norm = self.riemannian_squared_norm(ref_point, vector)
        norm = math.sqrt(sq_norm)
        return norm

    def riemannian_exp(self, ref_point, tangent_vec):
        """
        Compute the Riemannian exponential at point ref_point
        of tangent vector tangent_vec wrt the metric.
        """
        raise NotImplementedError(
                'The Riemannian exponential is not implemented.')

    def riemannian_log(self, ref_point, point):
        """
        Compute the Riemannian logarithm at point ref_point
        of tangent vector tangent_vec wrt the metric.
        """
        raise NotImplementedError(
                'The Riemannian logarithm is not implemented.')

    def riemannian_squared_dist(self, point_a, point_b):
        """
        Squared Riemannian distance between points
        point_a and point_b.

        Note: squared distance may be non-positive if the metric signature
        is not (0, dimension).
        """
        riem_log = self.riemannian_log(point_a, point_b)
        riem_sq_dist = self.riemannian_squared_norm(point_a, riem_log)
        return riem_sq_dist

    def riemannian_dist(self, point_a, point_b):
        """
        Riemannian distance between points
        point_a and point_b.

        Note: Only for Riemannian metrics, i.e. signature (0, dimension).
        """
        riem_sq_dist = self.riemannian_squared_distance(point_a, point_b)
        riem_dist = math.sqrt(riem_sq_dist)
        return riem_dist

    def random_uniform(self):
        """
        Samples a random point in this manifold according to
        the Riemannian measure.
        """
        raise NotImplementedError(
                'Uniform sampling w.r.t. Riemannian measure'
                'is not implemented.')

    def riemannian_variance(self, ref_point, points, weights):
        """
        Compute the weighted variance of the points
        in the tangent space at the ref_point.
        """
        n_points, _ = points.shape
        n_weights = len(weights)
        assert n_points > 0
        assert n_points == n_weights

        variance = 0

        for i in range(n_points):
            weight_i = weights[i]
            point_i = points[i, :]

            sq_geodesic_dist = self.riemannian_squared_distance(ref_point,
                                                                point_i)

            variance += weight_i * sq_geodesic_dist

        return variance

    def riemannian_mean(self, points, weights, epsilon=EPSILON):
        """
        Compute the weighted mean of the points.

        The geodesic distances are obtained with the
        Riemannian distance.
        """
        n_points, _ = points.shape
        n_weights = len(weights)
        assert n_points > 0
        assert n_points == n_weights

        riem_mean = points[0, :]

        if n_points == 1:
            return riem_mean

        while True:
            riem_mean_next = riem_mean
            aux = np.zeros(3)

            for i in range(n_points):
                point_i = points[i, :]
                weight_i = weights[i]

                aux += weight_i * self.riemannian_log(riem_mean_next, point_i)

            riem_mean = self.riemannian_exp(riem_mean_next, aux)

            diff = self.riemannian_squared_distance(riem_mean_next,
                                                    riem_mean)

            variance = self.riemannian_variance(riem_mean_next,
                                                points,
                                                weights)

            if diff < epsilon * variance:
                break

        return riem_mean
