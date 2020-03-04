"""Product of Riemannian metrics.

Define the metric of a product manifold endowed with a product metric.
"""

import geomstats.backend as gs
from geomstats.geometry.riemannian_metric import RiemannianMetric

EPSILON = 1e-5


# TODO(nina): unit tests

class ProductRiemannianMetric(RiemannianMetric):
    """Class for product of Riemannian metrics.

    Parameters
    ----------
    metrics : list
        List of metrics in the product.
    """

    def __init__(self, metrics):
        self.n_metrics = len(metrics)
        self.metrics = metrics
        dimensions = [metric.dimension for metric in metrics]
        self.dimensions = dimensions
        signatures = [metric.signature for metric in metrics]
        self.signatures = signatures

        sig_0 = sum([sig[0] for sig in signatures])
        sig_1 = sum([sig[1] for sig in signatures])
        sig_2 = sum([sig[2] for sig in signatures])
        super(ProductRiemannianMetric, self).__init__(
            dimension=sum(dimensions),
            signature=(sig_0, sig_1, sig_2))

    def inner_product_matrix(self, base_point=None):
        """Compute the matrix of the inner-product.

        Matrix of the inner-product defined by the Riemmanian metric
        at point base_point of the manifold.

        Parameters
        ----------
        base_point : array-like, shape=[n_samples, dimension], optional
            Point on the manifold at which to compute the inner-product matrix.

        Returns
        -------
        matrix : array-like, shape=[n_samples, dimension, dimension]
            Matrix of the inner-product at the base point.
        """
        matrix = gs.zeros([self.dimension, self.dimension])
        b = self.dimensions[0]
        matrix[:b, :b] = self.metrics.inner_product_matrix(base_point[0])
        dim_current = 0

        for i in range(self.n_metrics - 1):
            dim_current += self.dimensions[i]
            dim_next = self.dimensions[i + 1]
            a = dim_current
            b = dim_current + dim_next
            matrix_next = self.metrics.inner_product_matrix(base_point[i + 1])
            matrix[a:b, a:b] = matrix_next

        return matrix

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the inner-product of two tangent vectors at a base point.

        Inner product defined by the Riemannian metric at point `base_point`
        between tangent vectors `tangent_vec_a` and `tangent_vec_b`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[n_samples, dimension + 1]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[n_samples, dimension + 1]
            Second tangent vector at base point.
        base_point : array-like, shape=[n_samples, dimension + 1], optional
            Point on the manifold.

        Returns
        -------
        inner_prod : array-like, shape=[n_samples, 1]
            Inner-product of the two tangent vectors.
        """
        if base_point is None:
            base_point = [None, ] * self.n_metrics

        inner_products = [metric.inner_product(tangent_vec_a[i],
                                               tangent_vec_b[i],
                                               base_point[i])
                          for i, metric in enumerate(self.metrics)]
        inner_product = gs.sum(inner_products)

        return inner_product

    def exp(self, tangent_vec, base_point=None):
        """Compute Riemannian exponential of tangent vector at base point.

        Riemannian exponential at point base_point
        of tangent vector tangent_vec wrt the Riemannian metric.

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, dimension]
            Tangent vector at a base point.
        base_point : array-like, shape=[n_samples, dimension]
            Point on the manifold.

        Returns
        -------
        exp : array-like, shape=[n_samples, dimension]
            Point on the manifold equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        if base_point is None:
            base_point = [None, ] * self.n_metrics

        exp = gs.asarray([self.metrics[i].exp(tangent_vec[i],
                                              base_point[i])
                          for i in range(self.n_metrics)])
        return exp

    def log(self, point, base_point=None):
        """Compute the Riemannian logarithm of a point.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension]
            Point on the manifold.
        base_point : array-like, shape=[n_samples, dimension]
            Point on the manifold.

        Returns
        -------
        log : array-like, shape=[n_samples, dimension]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
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
        point_a : array-like, shape=[n_samples, dimension]
            First point on the manifold.
        point_b : array-like, shape=[n_samples, dimension]
            Second point on the manifold.

        Returns
        -------
        sq_dist : array-like, shape=[n_samples, 1]
            Geodesic distance between the two points.
        """
        sq_distances = gs.asarray(
            [metric.squared_dist(point_a[i], point_b[i])
             for i, metric in enumerate(self.metrics)])

        return sum(sq_distances)

    def geodesic(self, initial_point,
                 end_point=None, initial_tangent_vec=None,
                 point_type=None):
        """Compute the geodesic as a function of t.

        This geodesic is seen as the product of the geodesic on each space.
        The geodesic curve is defined by either:
        - an initial point and an initial tangent vector,
        or
        -an initial point and an end point.

        Parameters
        ----------
        initial_point : array-like, shape=[n_samples, dim]
            Initial point of the geodesic.
        end_point : array-like, shape=[n_samples, dim], optional
            End point of the geodesic.
        initial_tangent_vec : array-like, shape=[n_samples, dim], optional
            Initial tangent vector of the geodesic.
        point_type : str, {'vector', 'matrix'}, optional
            Representation of point.

        Returns
        -------
        path : callable
            The time-parameterized geodesic.
        """
        if point_type is None:
            point_type = self.default_point_type
        assert point_type in ['vector', 'matrix']

        if end_point is not None:
            geodesics = [
                metric.geodesic(
                    initial_point[:, i_space, ...],
                    end_point=end_point[:, i_space, ...],
                    point_type=point_type)
                for i_space, metric in enumerate(self.metrics)]
        else:
            geodesics = [
                metric.geodesic(
                    initial_point[:, i, ...],
                    initial_tangent_vec=initial_tangent_vec[:, i, ...],
                    point_type=point_type)
                for i, metric in enumerate(self.metrics)]

        def path(t):
            geod = gs.stack([geodesic(t) for geodesic in geodesics], axis=0)
            geod = gs.swapaxes(geod, 0, 1)
            return geod
        return path
