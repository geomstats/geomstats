"""Product of Riemannian metrics.

Define the metric of a product manifold endowed with a product metric.

Lead author: Nicolas Guigui.
"""

import joblib

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.riemannian_metric import RiemannianMetric


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

    def __init__(self, metrics, default_point_type="vector", n_jobs=1):

        geomstats.errors.check_parameter_accepted_values(
            default_point_type, "default_point_type", ["vector", "matrix"]
        )
        if default_point_type == "vector":
            shape = (sum([m.shape[0] for m in metrics]),)
        else:
            shape = (len(metrics), *metrics[0].shape)

        self.n_metrics = len(metrics)
        dims = [metric.dim for metric in metrics]
        signatures = [metric.signature for metric in metrics]

        sig_pos = sum(sig[0] for sig in signatures)
        sig_neg = sum(sig[1] for sig in signatures)
        super(ProductRiemannianMetric, self).__init__(
            dim=sum(dims),
            signature=(sig_pos, sig_neg),
            shape=shape,
        )

        self.metrics = metrics
        self.dims = dims
        self.signatures = signatures
        self.n_jobs = n_jobs

    def metric_matrix(self, base_point=None):
        """Compute the matrix of the inner-product.

        Matrix of the inner-product defined by the Riemmanian metric
        at point base_point of the manifold.

        Parameters
        ----------
        base_point : array-like, shape=[..., n_metrics, dim] or
            [..., dim]
            Point on the manifold at which to compute the inner-product matrix.
            Optional, default: None.

        Returns
        -------
        matrix : array-like, shape=[..., dim, dim] or
        [..., dim + n_metrics, dim + n_metrics]
            Matrix of the inner-product at the base point.

        """
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=3)

        matrix = gs.zeros([len(base_point), self.dim, self.dim])
        cum_dim = 0
        for i in range(self.n_metrics):
            cum_dim_next = cum_dim + self.dims[i]
            if self.default_point_type == "vector":
                matrix_next = self.metrics[i].metric_matrix(
                    base_point[:, cum_dim:cum_dim_next, cum_dim:cum_dim_next]
                )
            else:
                matrix_next = self.metrics[i].metric_matrix(base_point[:, i])

            matrix[:, cum_dim:cum_dim_next, cum_dim:cum_dim_next] = matrix_next
            cum_dim = cum_dim_next
        return matrix[0] if len(base_point) == 1 else matrix

    def is_intrinsic(self, point):
        """Test in a point is represented in intrinsic coordinates.

        This method is only useful for `point_type == vector`.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point on the product manifold.

        Returns
        -------
        intrinsic : array-like, shape=[...,]
            Whether intrinsic coordinates are used for all manifolds.
        """
        if self.default_point_type != "vector":
            raise ValueError("Invalid default_point_type: 'vector' expected.")

        if point.shape[-1] == self.dim:
            return True
        if point.shape[-1] == sum(dim + 1 for dim in self.dims):
            return False
        raise ValueError("Input shape does not match the dimension of the manifold")

    @staticmethod
    def _get_method(metric, method_name, metric_args):
        return getattr(metric, method_name)(**metric_args)

    def _iterate_over_metrics(self, func, args, intrinsic=False):

        cum_index = (
            gs.cumsum(self.dims)[:-1]
            if intrinsic
            else gs.cumsum(gs.array([k + 1 for k in self.dims]))
        )
        arguments = {
            key: gs.split(args[key], cum_index, axis=-1) for key in args.keys()
        }
        args_list = [
            {key: arguments[key][j] for key in args.keys()}
            for j in range(self.n_metrics)
        ]
        pool = joblib.Parallel(n_jobs=self.n_jobs, prefer="threads")
        out = pool(
            joblib.delayed(self._get_method)(self.metrics[i], func, args_list[i])
            for i in range(self.n_metrics)
        )
        return out

    def inner_product(
        self,
        tangent_vec_a,
        tangent_vec_b,
        base_point=None,
    ):
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

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Inner-product of the two tangent vectors.
        """
        if base_point is None:
            base_point = gs.empty((self.n_metrics, self.dim))

        if self.default_point_type == "vector":
            intrinsic = self.is_intrinsic(tangent_vec_b)
            args = {
                "tangent_vec_a": tangent_vec_a,
                "tangent_vec_b": tangent_vec_b,
                "base_point": base_point,
            }
            inner_prod = self._iterate_over_metrics("inner_product", args, intrinsic)
            return gs.sum(gs.stack(inner_prod, axis=-2), axis=-2)

        inner_products = [
            metric.inner_product(
                tangent_vec_a[..., i, :],
                tangent_vec_b[..., i, :],
                base_point[..., i, :],
            )
            for i, metric in enumerate(self.metrics)
        ]
        return sum(inner_products)

    def exp(self, tangent_vec, base_point=None, **kwargs):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
            Optional, default: None.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            Point on the manifold equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        if base_point is None:
            base_point = [
                None,
            ] * self.n_metrics

        if self.default_point_type == "vector":
            intrinsic = self.is_intrinsic(base_point)
            args = {"tangent_vec": tangent_vec, "base_point": base_point}
            exp = self._iterate_over_metrics("exp", args, intrinsic)
            return gs.concatenate(exp, -1)

        exp = gs.stack(
            [
                self.metrics[i].exp(tangent_vec[..., i, :], base_point[..., i, :])
                for i in range(self.n_metrics)
            ],
            axis=-2,
        )
        return exp[0] if len(tangent_vec) == 1 else exp

    def log(self, point, base_point=None, **kwargs):
        """Compute the Riemannian logarithm of a point.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point on the manifold.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
            Optional, default: None.

        Returns
        -------
        log : array-like, shape=[..., dim]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        if base_point is None:
            base_point = [None] * self.n_metrics

        if self.default_point_type == "vector":
            intrinsic = self.is_intrinsic(base_point)
            args = {"point": point, "base_point": base_point}
            logs = self._iterate_over_metrics("log", args, intrinsic)
            logs = gs.concatenate(logs, axis=-1)
            return logs

        logs = gs.stack(
            [
                self.metrics[i].log(point[..., i, :], base_point[..., i, :])
                for i in range(self.n_metrics)
            ],
            axis=-2,
        )
        return logs


class NFoldMetric(RiemannianMetric):
    r"""Class for an n-fold product manifold :math:`M^n`.

    Define a manifold as the product manifold of n copies of a given base
    manifold M.

    Parameters
    ----------
    base_metric : RiemannianMetric
        Base metric.
    n_copies : int
        Number of replication of the base metric.
    """

    def __init__(self, base_metric, n_copies):
        geomstats.errors.check_integer(n_copies, "n_copies")
        dim = n_copies * base_metric.dim
        base_shape = base_metric.shape
        super(NFoldMetric, self).__init__(dim=dim, shape=(n_copies, *base_shape))
        self.base_shape = base_shape
        self.base_metric = base_metric
        self.n_copies = n_copies

    def metric_matrix(self, base_point=None):
        """Compute the matrix of the inner-product.

        Matrix of the inner-product defined by the Riemmanian metric
        at point base_point of the manifold.

        Parameters
        ----------
        base_point : array-like, shape=[..., n_copies, *base_shape]
            Point on the manifold at which to compute the inner-product matrix.
            Optional, default: None.

        Returns
        -------
        matrix : array-like, shape=[..., n_copies, dim, dim]
            Matrix of the inner-product at the base point.
        """
        point_ = gs.reshape(base_point, (-1, *self.base_shape))
        matrices = self.base_metric.metric_matrix(point_)
        dim = self.base_metric.dim
        reshaped = gs.reshape(matrices, (-1, self.n_copies, dim, dim))
        return gs.squeeze(reshaped)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the inner-product of two tangent vectors at a base point.

        Inner product defined by the Riemannian metric at point `base_point`
        between tangent vectors `tangent_vec_a` and `tangent_vec_b`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., n_copies, *base_shape]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., n_copies, *base_shape]
            Second tangent vector at base point.
        base_point : array-like, shape=[..., n_copies, *base_shape]
            Point on the manifold.
            Optional, default: None.

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Inner-product of the two tangent vectors.
        """
        tangent_vec_a_, tangent_vec_b_, point_ = gs.broadcast_arrays(
            tangent_vec_a, tangent_vec_b, base_point
        )
        point_ = gs.reshape(point_, (-1, *self.base_shape))
        vector_a = gs.reshape(tangent_vec_a_, (-1, *self.base_shape))
        vector_b = gs.reshape(tangent_vec_b_, (-1, *self.base_shape))
        inner_each = self.base_metric.inner_product(vector_a, vector_b, point_)
        reshaped = gs.reshape(inner_each, (-1, self.n_copies))
        return gs.squeeze(gs.sum(reshaped, axis=-1))

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n_copies, *base_shape]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., n_copies, *base_shape]
            Point on the manifold.
            Optional, default: None.

        Returns
        -------
        exp : array-like, shape=[..., n_copies, *base_shape]
            Point on the manifold equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        tangent_vec, point_ = gs.broadcast_arrays(tangent_vec, base_point)
        point_ = gs.reshape(point_, (-1, *self.base_shape))
        vector_ = gs.reshape(tangent_vec, (-1, *self.base_shape))
        each_exp = self.base_metric.exp(vector_, point_)
        reshaped = gs.reshape(each_exp, (-1, self.n_copies) + self.base_shape)
        return gs.squeeze(reshaped)

    def log(self, point, base_point, **kwargs):
        """Compute the Riemannian logarithm of a point.

        Parameters
        ----------
        point : array-like, shape=[..., n_copies, *base_shape]
            Point on the manifold.
        base_point : array-like, shape=[..., n_copies, *base_shape]
            Point on the manifold.
            Optional, default: None.

        Returns
        -------
        log : array-like, shape=[..., n_copies, *base_shape]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        point_, base_point_ = gs.broadcast_arrays(point, base_point)
        base_point_ = gs.reshape(base_point_, (-1, *self.base_shape))
        point_ = gs.reshape(point_, (-1, *self.base_shape))
        each_log = self.base_metric.log(point_, base_point_)
        reshaped = gs.reshape(each_log, (-1, self.n_copies) + self.base_shape)
        return gs.squeeze(reshaped)

    def geodesic(self, initial_point, end_point=None, initial_tangent_vec=None):
        """Generate parameterized function for the geodesic curve.

        Geodesic curve defined by either:

        - an initial landmark set and an initial tangent vector,
        - an initial landmark set and an end landmark set.

        Parameters
        ----------
        initial_point : array-like, shape=[..., dim]
            Landmark set, initial point of the geodesic.
        end_point : array-like, shape=[..., dim]
            Landmark set, end point of the geodesic. If None,
            an initial tangent vector must be given.
            Optional, default : None
        initial_tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point, the initial speed of the geodesics.
            If None, an end point must be given and a logarithm is computed.
            Optional, default : None

        Returns
        -------
        path : callable
            Time parameterized geodesic curve.
        """
        if end_point is None and initial_tangent_vec is None:
            raise ValueError(
                "Specify an end landmark set or an initial tangent"
                "vector to define the geodesic."
            )
        if end_point is not None:
            shooting_tangent_vec = self.log(point=end_point, base_point=initial_point)
            if initial_tangent_vec is not None:
                if not gs.allclose(shooting_tangent_vec, initial_tangent_vec):
                    raise RuntimeError(
                        "The shooting tangent vector is too"
                        " far from the initial tangent vector."
                    )
            initial_tangent_vec = shooting_tangent_vec
        initial_tangent_vec = gs.array(initial_tangent_vec)

        def landmarks_on_geodesic(t):
            t = gs.cast(t, initial_point.dtype)
            t = gs.to_ndarray(t, to_ndim=1)

            tangent_vecs = gs.einsum("...,...ij->...ij", t, initial_tangent_vec)

            def point_ok_landmarks(tangent_vec):
                if gs.ndim(tangent_vec) < 2:
                    raise RuntimeError
                exp = self.exp(tangent_vec=tangent_vec, base_point=initial_point)
                return exp

            landmarks_at_time_t = gs.vectorize(
                tangent_vecs, point_ok_landmarks, signature="(i,j)->(i,j)"
            )

            return landmarks_at_time_t

        return landmarks_on_geodesic
