"""Product of Riemannian metrics.

Define the metric of a product manifold endowed with a product metric.
"""

import joblib

import geomstats.backend as gs
import geomstats.errors
import geomstats.vectorization
from geomstats.geometry.riemannian_metric import RiemannianMetric

EPSILON = 1e-5


class ProductRiemannianMetric(RiemannianMetric):
    """Class for product of Riemannian metrics.

    Parameters
    ----------
    metrics : list
        List of metrics in the product.
    default_point_type : str, {'vector', 'matrix'}
        Point type.
        Optional, default: 'vector'.
    n_jobs : int
        Number of jobs for parallel computing.
        Optional, default: 1.
    """

    def __init__(self, metrics, default_point_type='vector', n_jobs=1):
        self.n_metrics = len(metrics)
        dims = [metric.dim for metric in metrics]
        signatures = [metric.signature for metric in metrics]

        sig_0 = sum(sig[0] for sig in signatures)
        sig_1 = sum(sig[1] for sig in signatures)
        sig_2 = sum(sig[2] for sig in signatures)
        super(ProductRiemannianMetric, self).__init__(
            dim=sum(dims),
            signature=(sig_0, sig_1, sig_2),
            default_point_type=default_point_type)

        self.metrics = metrics
        self.dims = dims
        self.signatures = signatures
        self.n_jobs = n_jobs

    def inner_product_matrix(self, base_point=None, point_type=None):
        """Compute the matrix of the inner-product.

        Matrix of the inner-product defined by the Riemmanian metric
        at point base_point of the manifold.

        Parameters
        ----------
        base_point : array-like, shape=[..., n_metrics, dim] or
            [..., dim]
            Point on the manifold at which to compute the inner-product matrix.
            Optional, default: None.
        point_type : str, {'vector', 'matrix'}
            Type of representation used for points.
            Optional, default: None.

        Returns
        -------
        matrix : array-like, shape=[..., dim, dim] or
        [..., dim + n_metrics, dim + n_metrics]
            Matrix of the inner-product at the base point.

        """
        if point_type is None:
            point_type = self.default_point_type

        base_point = gs.to_ndarray(base_point, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=3)

        matrix = gs.zeros([
            len(base_point), self.dim, self.dim])
        cum_dim = 0
        for i in range(self.n_metrics):
            cum_dim_next = cum_dim + self.dims[i]
            if point_type == 'matrix':
                matrix_next = self.metrics[i].inner_product_matrix(
                    base_point[:, i])
            elif point_type == 'vector':
                matrix_next = self.metrics[i].inner_product_matrix(
                    base_point[:, cum_dim:cum_dim_next, cum_dim:cum_dim_next])
            else:
                raise ValueError('invalid point_type argument: {}, expected '
                                 'either matrix of vector'.format(point_type))
            matrix[:, cum_dim:cum_dim_next, cum_dim:cum_dim_next] = matrix_next
            cum_dim = cum_dim_next
        return matrix[0] if len(base_point) == 1 else matrix

    def is_intrinsic(self, point):
        """Test in a point is represented in intrinsic coordinates.

        This method is only useful for `point_type=vector`.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point on the product manifold.

        Returns
        -------
        intrinsic : array-like, shape=[...,]
            Whether intrinsic coordinates are used for all manifolds.
        """
        if self.default_point_type != 'vector':
            raise ValueError(
                'Invalid default_point_type: \'vector\' expected.')

        if point.shape[1] == self.dim:
            intrinsic = True
        elif point.shape[1] == sum(dim + 1 for dim in self.dims):
            intrinsic = False
        else:
            raise ValueError(
                'Input shape does not match the dimension of the manifold')
        return intrinsic

    @staticmethod
    def _get_method(metric, method_name, metric_args):
        return getattr(metric, method_name)(**metric_args)

    def _iterate_over_metrics(
            self, func, args, intrinsic=False):

        cum_index = gs.cumsum(self.dims, axis=0)[:-1] if intrinsic else \
            gs.cumsum(gs.array([k + 1 for k in self.dims]), axis=0)
        arguments = {
            key: gs.split(args[key], cum_index, axis=1) for key in args.keys()}
        args_list = [{key: arguments[key][j] for key in args.keys()} for j in
                     range(self.n_metrics)]
        pool = joblib.Parallel(n_jobs=self.n_jobs, prefer='threads')
        out = pool(
            joblib.delayed(self._get_method)(
                self.metrics[i], func, args_list[i]) for i in range(
                self.n_metrics))
        return out

    def inner_product(
            self, tangent_vec_a, tangent_vec_b, base_point=None,
            point_type=None):
        """Compute the inner-product of two tangent vectors at a base point.

        Inner product defined by the Riemannian metric at point `base_point`
        between tangent vectors `tangent_vec_a` and `tangent_vec_b`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim + 1]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., dim + 1]
            Second tangent vector at base point.
        base_point : array-like, shape=[..., dim + 1]
            Point on the manifold.
            Optional, default: None.
        point_type : str, {'vector', 'matrix'}
            Type of representation used for points.
            Optional, default: None.

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Inner-product of the two tangent vectors.
        """
        if base_point is None:
            base_point = gs.empty((self.n_metrics, self.dim))

        if point_type is None:
            point_type = self.default_point_type
        geomstats.errors.check_parameter_accepted_values(
            point_type, 'point_type', ['vector', 'matrix'])

        tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=2)
        tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        if point_type == 'vector':
            intrinsic = self.is_intrinsic(tangent_vec_b)
            args = {'tangent_vec_a': tangent_vec_a,
                    'tangent_vec_b': tangent_vec_b,
                    'base_point': base_point}
            inner_prod = self._iterate_over_metrics(
                'inner_product', args, intrinsic)
            return gs.sum(gs.stack(inner_prod, axis=1), axis=1)

        inner_products = [
            metric.inner_product(
                tangent_vec_a[..., i, :],
                tangent_vec_b[..., i, :],
                base_point[..., i, :])
            for i, metric in enumerate(self.metrics)]
        return sum(inner_products)

    def exp(self, tangent_vec, base_point=None, point_type=None):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
            Optional, default: None.
        point_type : str, {'vector', 'matrix'}
            Type of representation used for points.
            Optional, default: None.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            Point on the manifold equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        if base_point is None:
            base_point = [None, ] * self.n_metrics

        if point_type is None:
            point_type = self.default_point_type
        geomstats.errors.check_parameter_accepted_values(
            point_type, 'point_type', ['vector', 'matrix'])

        if point_type == 'vector':
            tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
            base_point = gs.to_ndarray(base_point, to_ndim=2)
            intrinsic = self.is_intrinsic(base_point)
            args = {'tangent_vec': tangent_vec, 'base_point': base_point}
            exp = self._iterate_over_metrics('exp', args, intrinsic)
            return gs.hstack(exp)

        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
        base_point = gs.to_ndarray(base_point, to_ndim=3)
        exp = gs.stack([
            self.metrics[i].exp(tangent_vec[:, i], base_point[:, i])
            for i in range(self.n_metrics)], axis=1)
        return exp[0] if len(tangent_vec) == 1 else exp

    def log(self, point, base_point=None, point_type=None):
        """Compute the Riemannian logarithm of a point.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point on the manifold.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
            Optional, default: None.
        point_type : str, {'vector', 'matrix'}
            Type of representation used for points.
            Optional, default: None.

        Returns
        -------
        log : array-like, shape=[..., dim]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        if base_point is None:
            base_point = [None] * self.n_metrics

        if point_type is None:
            point_type = self.default_point_type
        geomstats.errors.check_parameter_accepted_values(
            point_type, 'point_type', ['vector', 'matrix'])

        if point_type == 'vector':
            point = gs.to_ndarray(point, to_ndim=2)
            base_point = gs.to_ndarray(base_point, to_ndim=2)
            intrinsic = self.is_intrinsic(base_point)
            args = {'point': point, 'base_point': base_point}
            logs = self._iterate_over_metrics('log', args, intrinsic)
            logs = gs.concatenate(logs, axis=1)
            return logs

        point = gs.to_ndarray(point, to_ndim=2, axis=0)
        point = gs.to_ndarray(point, to_ndim=3, axis=0)
        base_point = gs.to_ndarray(base_point, to_ndim=2, axis=0)
        base_point = gs.to_ndarray(base_point, to_ndim=3, axis=0)
        logs = gs.stack(
            [self.metrics[i].log(point[:, i], base_point[:, i])
             for i in range(self.n_metrics)], axis=1)
        return logs
