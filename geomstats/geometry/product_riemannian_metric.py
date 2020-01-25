"""Product of Riemannian metrics."""

import geomstats.backend as gs
from geomstats.geometry.riemannian_metric import RiemannianMetric

EPSILON = 1e-5


# TODO(nina): unit tests

class ProductRiemannianMetric(RiemannianMetric):
    """Class for product of Riemannian metrics."""

    def __init__(self, metrics):
        self.n_metrics = len(metrics)
        dimensions = [metric.dimension for metric in metrics]
        signatures = [metric.signature for metric in metrics]

        self.metrics = metrics
        self.dimensions = dimensions
        self.signatures = signatures

        sig_0 = sum([sig[0] for sig in signatures])
        sig_1 = sum([sig[1] for sig in signatures])
        sig_2 = sum([sig[2] for sig in signatures])
        super(ProductRiemannianMetric, self).__init__(
            dimension=sum(dimensions),
            signature=(sig_0, sig_1, sig_2))

    def inner_product_matrix(self, base_point=None):
        """Compute matrix of the corresponding inner product.

        Matrix of the inner product defined by the Riemmanian metric
        at point base_point of the manifold.

        Parameters
        ----------
        base_point

        Returns
        -------
        matrix
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
        """Compute inner product between two tan space vectors at a base point.

        Inner product defined by the Riemannian metric at point `base_point`
        between tangent vectors `tangent_vec_a` and `tangent_vec_b`.

        Parameters
        ----------
        tangent_vec_a
        tangent_vec_b
        base_point

        Returns
        -------
        inner_product
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
        """Compute Riemannian exponential of tangent vector at base point.

        Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the Riemannian metric.

        Parameters
        ----------
        tangent_vec
        base_point

        Returns
        -------
        exp
        """
        if base_point is None:
            base_point = [None, ] * self.n_metrics

        exp = gs.asarray([self.metrics[i].exp(tangent_vec[i],
                                              base_point[i])
                          for i in range(self.n_metrics)])
        return exp

    def log(self, point, base_point=None):
        """Compute Riemannian logarithm of a point wrt a base point.

        Parameters
        ----------
        point
        base_point

        Returns
        -------
        log
        """
        if base_point is None:
            base_point = [None, ] * self.n_metrics

        log = gs.asarray([self.metrics[i].log(point[i],
                                              base_point[i])
                          for i in range(self.n_metrics)])
        return log

    def squared_dist(self, point_a, point_b):
        """Compute squared geodesic distance between two points.

        Parameters
        ----------
        point_a: array-like, shape=[n_samples, dimension]
                             or shape=[1, dimension]
        point_b: array-like, shape=[n_samples, dimension]
                             or shape=[1, dimension]

        Returns
        -------
        sum_sq_distances
        """
        sq_distances = gs.asarray(
            [self.metrics[i].squared_dist(
                point_a[i], point_b[i])
             for i in range(self.n_metrics)])

        return sum(sq_distances)
