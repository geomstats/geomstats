"""Product of Riemannian metrics."""

import multiprocessing as mp

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

    def __init__(self, metrics, default_point_type='vector', n_jobs=1):
        self.n_metrics = len(metrics)
        dimensions = [metric.dimension for metric in metrics]
        signatures = [metric.signature for metric in metrics]

        self.metrics = metrics
        self.dimensions = dimensions
        self.signatures = signatures
        self.default_point_type = default_point_type
        self.n_jobs = n_jobs

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

    def _detect_intrinsic_extrinsic(self, point, point_type):
        assert point_type in ['vector', 'matrix']
        index = 1 if point_type == 'vector' else 2
        if point.shape[index] == self.dimension:
            intrinsic = True
        elif point.shape[index] == sum(
                [dim + 1 for dim in self.dimensions]):
            intrinsic = False
        else:
            raise ValueError('Input shape does not match the dimension of'
                             'the manifold, {0} expected {1} or {2}'.format(
                              point.shape, self.dimension,  sum(
                              [dim + 1 for dim in self.dimensions])))
        return intrinsic

    @staticmethod
    def _get_method(metric, method_name, metric_args):
        return getattr(metric, method_name)(**metric_args)

    def _iterate_over_metrics(
            self, func, args, intrinsic=False):

        cum_index = gs.cumsum(self.dimensions)[:-1] if intrinsic else \
            gs.cumsum([k + 1 for k in self.dimensions])
        arguments = {key: gs.split(
            args[key], cum_index, axis=1) for key in args.keys()}
        args_list = [{key: arguments[key][j] for key in args.keys()} for j in
                     range(self.n_metrics)]
        pool = mp.Pool(min(self.n_jobs, mp.cpu_count()))
        out = pool.starmap(
            self._get_method,
            [(self.metrics[i], func, args_list[i]) for i in range(
                self.n_metrics)])
        pool.close()
        return out

    def inner_product(
            self, tangent_vec_a, tangent_vec_b, base_point=None,
            point_type=None):
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

        if point_type is None:
            point_type = self.default_point_type
        if point_type == 'vector':
            tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=2)
            tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=2)
            base_point = gs.to_ndarray(base_point, to_ndim=2)
        elif point_type == 'matrix':
            tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=3)
            tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=3)
            base_point = gs.to_ndarray(base_point, to_ndim=3)
        intrinsic = self._detect_intrinsic_extrinsic(tangent_vec_b, point_type)
        args = {'tangent_vec_a': tangent_vec_a,
                'tangent_vec_b': tangent_vec_b,
                'base_point': base_point}
        inner_prod = self._iterate_over_metrics(
            'inner_product', args, intrinsic)
        return gs.sum(gs.hstack(inner_prod), axis=1)

    def exp(self, tangent_vec, base_point=None, point_type=None):
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

        if point_type is None:
            point_type = self.default_point_type
        if point_type == 'vector':
            tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
            base_point = gs.to_ndarray(base_point, to_ndim=2)
        elif point_type == 'matrix':
            tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
            base_point = gs.to_ndarray(base_point, to_ndim=3)
        intrinsic = self._detect_intrinsic_extrinsic(base_point, point_type)
        args = {'tangent_vec': tangent_vec, 'base_point': base_point}
        exp = self._iterate_over_metrics('exp', args, intrinsic)
        return gs.hstack(exp)

    def log(self, point, base_point=None, point_type=None):
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

        if point_type is None:
            point_type = self.default_point_type
        if point_type == 'vector':
            point = gs.to_ndarray(point, to_ndim=2)
            base_point = gs.to_ndarray(base_point, to_ndim=2)
        elif point_type == 'matrix':
            point = gs.to_ndarray(point, to_ndim=3)
            base_point = gs.to_ndarray(base_point, to_ndim=3)
        intrinsic = self._detect_intrinsic_extrinsic(base_point, point_type)
        args = {'point': point, 'base_point': base_point}
        log = self._iterate_over_metrics('log', args, intrinsic)
        return gs.hstack(log)
