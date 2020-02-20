"""Product of Riemannian metrics."""

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

        # FIXME: this assumes point_type is vector
        tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=2)
        tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=2)

        # detect if intrinsic of extrinsic
        if tangent_vec_a.shape[1] == self.dimension:
            intrinsic = True
        elif tangent_vec_a.shape[1] == sum([k + 1 for k in self.dimensions]):
            intrinsic = False
        else:
            raise ValueError('invalid input dimension')

        inner_prod = gs.zeros((len(tangent_vec_a), len(self.metrics)))
        cum_dim = 0
        for i, metric_i in enumerate(self.metrics):
            cum_dim_next = cum_dim + self.dimensions[i]
            if not intrinsic:
                cum_dim_next += 1
            tangent_vec_a_i = tangent_vec_a[:, cum_dim:cum_dim_next]
            tangent_vec_b_i = tangent_vec_b[:, cum_dim:cum_dim_next]
            base_point_i = base_point[:, cum_dim:cum_dim_next]
            inner_prod[:, i] = metric_i.inner_product(
                tangent_vec_a_i, tangent_vec_b_i, base_point_i).flatten()
            cum_dim = cum_dim_next

        return gs.sum(inner_prod, axis=1)

    def exp(self, tangent_vec, base_point=None):
        """Compute the Riemannian exponential of a tangent vector.

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

        # FIXME: this assumes point_type is vector
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
        # detect if intrinsic of extrinsic
        if tangent_vec.shape[1] == self.dimension:
            intrinsic = True
        elif tangent_vec.shape[1] == sum([k + 1 for k in self.dimensions]):
            intrinsic = False
        else:
            raise ValueError('invalid input dimension')

        exp = gs.zeros_like(tangent_vec)
        cum_dim = 0
        for i, metric_i in enumerate(self.metrics):
            cum_dim_next = cum_dim + self.dimensions[i]
            if not intrinsic:
                cum_dim_next += 1
            tangent_vec_i = tangent_vec[:, cum_dim:cum_dim_next]
            base_point_i = base_point[:, cum_dim:cum_dim_next]
            exp[:, cum_dim:cum_dim_next] = metric_i.exp(
                tangent_vec_i, base_point_i)
            cum_dim = cum_dim_next

        return exp

    def log(self, point, base_point=None):
        """Compute the Riemannian logarithm of a point.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension]
            Point on the manifold
        base_point : array-like, shape=[n_samples, dimension]
            Point on the manifold

        Returns
        -------
        log : array-like, shape=[n_samples, dimension]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        if base_point is None:
            base_point = [None, ] * self.n_metrics

        base_point = gs.to_ndarray(base_point, to_ndim=2)
        # FIXME: this assumes point_type is vector
        point = gs.to_ndarray(point, to_ndim=2)
        # detect if intrinsic of extrinsic
        if point.shape[1] == self.dimension:
            intrinsic = True
        elif point.shape[1] == sum([k + 1 for k in self.dimensions]):
            intrinsic = False
        else:
            raise ValueError('invalid input dimension')

        log = gs.zeros_like(point)
        cum_dim = 0
        for i, metric_i in enumerate(self.metrics):
            cum_dim_next = cum_dim + self.dimensions[i]
            if not intrinsic:
                cum_dim_next += 1
            point_i = point[:, cum_dim:cum_dim_next]
            base_point_i = base_point[:, cum_dim:cum_dim_next]
            log[:, cum_dim:cum_dim_next] = metric_i.log(
                point_i, base_point_i)
            cum_dim = cum_dim_next

        return log
