"""
Product of Riemannian metrics.
"""

import geomstats.backend as gs

from geomstats.riemannian_metric import RiemannianMetric

EPSILON = 1e-5


# TODO(nina): unit tests

class ProductRiemannianMetric(RiemannianMetric):
    """
    Class for product of Riemannian metrics.
    """

    def __init__(self, metrics):
        self.n_metrics = gs.len(metrics)
        dimensions = [metric.dimension for metric in metrics]
        signatures = [metric.signature for metric in metrics]

        self.metrics = metrics
        self.dimensions = dimensions
        self.signatures = signatures
        super(ProductRiemannianMetric, self).__init__(
            dimension=gs.sum(dimensions),
            signature=map(sum, signatures))

    def inner_product_matrix(self, base_point=None):
        """
        Matrix of the inner product defined by the Riemmanian metric
        at point base_point of the manifold.
        """
        matrix = gs.zeros([self.dimension, self.dimension])
        b = self.dimensions[0]
        matrix[:b, :b] = self.metrics.inner_product_matrix(base_point[0])
        dim_current = 0

        for i in range(self.n_metrics-1):
            dim_current += self.dimensions[i]
            dim_next = self.dimensions[i+1]
            a = dim_current
            b = dim_current + dim_next
            matrix_next = self.metrics.inner_product_matrix(base_point[i+1])
            matrix[a:b, a:b] = matrix_next

        return matrix

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """
        Inner product defined by the Riemannian metric at point base_point
        between tangent vectors tangent_vec_a and tangent_vec_b.
        """
        if base_point is None:
            base_point = [None, ] * self.n_metrics

        inner_products = [self.metrics[i].inner_product(tangent_vec_a[i],
                                                        tangent_vec_b[i],
                                                        base_point[i])
                          for i in range(self.n_metrics)]
        inner_product = gs.sum(inner_products)
        return inner_product

    def exp(self, tangent_vec, base_point=None):
        """
        Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the Riemannian metric.
        """
        if base_point is None:
            base_point = [None, ] * self.n_metrics

        exp = [self.metrics[i].exp(tangent_vec[i], base_point[i])
               for i in range(self.n_metrics)]

        return exp

    def log(self, point, base_point=None):
        """
        Riemannian logarithm at point base_point
        of tangent vector tangent_vec wrt the Riemannian metric.
        """
        if base_point is None:
            base_point = [None, ] * self.n_metrics

        log = [self.metrics[i].log(point[i], base_point[i])
               for i in range(self.n_metrics)]

        return log
