"""Product of manifolds.

Lead author: Nicolas Guigui.
"""

import joblib

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.product_riemannian_metric import ProductRiemannianMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric


class ProductManifold(Manifold):
    """Class for a product of manifolds M_1 x ... x M_n.

    In contrast to the classes NFoldManifold, Landmarks, or DiscretizedCurves,
    the manifolds M_1, ..., M_n need not be the same, nor of
    same dimension, but the list of manifolds needs to be provided.

    By default, a point is represented by an array of shape:
    [..., dim_1 + ... + dim_n_manifolds]
    where n_manifolds is the number of manifolds in the product.
    This type of representation is called 'vector'.

    Alternatively, a point can be represented by an array of shape:
    [..., n_manifolds, dim] if the n_manifolds have same dimension dim.
    This type of representation is called `matrix`.

    Parameters
    ----------
    manifolds : list
        List of manifolds in the product.
    default_point_type : str, {'vector', 'matrix'}
        Default representation of points.
        Optional, default: 'vector'.
    n_jobs : int
        Number of jobs for parallel computing.
        Optional, default: 1.
    """

    # FIXME (nguigs): This only works for 1d points

    def __init__(
        self, manifolds, metrics=None, default_point_type="vector", n_jobs=1, **kwargs
    ):
        geomstats.errors.check_parameter_accepted_values(
            default_point_type, "default_point_type", ["vector", "matrix"]
        )

        self.dims = [manifold.dim for manifold in manifolds]
        if metrics is None:
            metrics = [manifold.metric for manifold in manifolds]
        metric = ProductRiemannianMetric(
            metrics, default_point_type=default_point_type, n_jobs=n_jobs
        )
        dim = sum(self.dims)

        if default_point_type == "vector":
            shape = (sum([m.shape[0] for m in manifolds]),)
        else:
            shape = (len(manifolds), *manifolds[0].shape)

        super(ProductManifold, self).__init__(
            dim=dim,
            shape=shape,
            metric=metric,
            default_point_type=default_point_type,
            **kwargs,
        )
        self.manifolds = manifolds
        self.n_jobs = n_jobs

    @staticmethod
    def _get_method(manifold, method_name, metric_args):
        return getattr(manifold, method_name)(**metric_args)

    def _iterate_over_manifolds(self, func, args, intrinsic=False):

        cum_index = (
            gs.cumsum(self.dims)[:-1]
            if intrinsic
            else gs.cumsum([k + 1 for k in self.dims])
        )
        arguments = {}
        float_args = {}
        for key, value in args.items():
            if not isinstance(value, float):
                arguments[key] = gs.split(value, cum_index, axis=-1)
            else:
                float_args[key] = value
        args_list = [
            {key: arguments[key][j] for key in arguments}
            for j in range(len(self.manifolds))
        ]
        pool = joblib.Parallel(n_jobs=self.n_jobs)
        out = pool(
            joblib.delayed(self._get_method)(
                self.manifolds[i], func, {**args_list[i], **float_args}
            )
            for i in range(len(self.manifolds))
        )
        return out

    def belongs(self, point, atol=gs.atol):
        """Test if a point belongs to the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n_manifolds, dim_each]}]
            Point.
        atol : float,
            Tolerance.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if the point belongs to the manifold.
        """
        point_type = self.default_point_type

        if point_type == "vector":
            intrinsic = self.metric.is_intrinsic(point)
            belongs = self._iterate_over_manifolds(
                "belongs", {"point": point, "atol": atol}, intrinsic
            )
            belongs = gs.stack(belongs, axis=-1)

        else:
            belongs = gs.stack(
                [
                    space.belongs(point[..., i, :], atol)
                    for i, space in enumerate(self.manifolds)
                ],
                axis=-1,
            )

        belongs = gs.all(belongs, axis=-1)
        return belongs

    def regularize(self, point):
        """Regularize the point into the manifold's canonical representation.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n_manifolds, dim_each]}]
            Point to be regularized.
        point_type : str, {'vector', 'matrix'}
            Representation of point.
            Optional, default: None.

        Returns
        -------
        regularized_point : array-like,
            shape=[..., {dim, [n_manifolds, dim_each]}]
            Point in the manifold's canonical representation.
        """
        point_type = self.default_point_type

        if point_type == "vector":
            intrinsic = self.metric.is_intrinsic(point)
            regularized_point = self._iterate_over_manifolds(
                "regularize", {"point": point}, intrinsic
            )
            regularized_point = gs.concatenate(regularized_point, axis=-1)
        elif point_type == "matrix":
            regularized_point = [
                manifold_i.regularize(point[..., i, :])
                for i, manifold_i in enumerate(self.manifolds)
            ]
            regularized_point = gs.stack(regularized_point, axis=1)
        return regularized_point

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in the product space from the uniform distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples.
        bound : float
            Bound of the interval in which to sample for non compact manifolds.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., {dim, [n_manifolds, dim_each]}]
            Points sampled on the hypersphere.
        """
        point_type = self.default_point_type
        geomstats.errors.check_parameter_accepted_values(
            point_type, "point_type", ["vector", "matrix"]
        )

        if point_type == "vector":
            data = self.manifolds[0].random_point(n_samples, bound)
            if len(self.manifolds) > 1:
                for space in self.manifolds[1:]:
                    samples = space.random_point(n_samples, bound)
                    data = gs.concatenate([data, samples], axis=-1)
            return data

        point = [space.random_point(n_samples, bound) for space in self.manifolds]
        samples = gs.stack(point, axis=-2)
        return samples

    def projection(self, point):
        """Project a point in product embedding manifold on each manifold.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n_manifolds, dim_each]}]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., {dim, [n_manifolds, dim_each]}]
            Projected point.
        """
        point_type = self.default_point_type
        geomstats.errors.check_parameter_accepted_values(
            point_type, "point_type", ["vector", "matrix"]
        )

        if point_type == "vector":
            intrinsic = self.metric.is_intrinsic(point)
            projected_point = self._iterate_over_manifolds(
                "projection", {"point": point}, intrinsic
            )
            projected_point = gs.concatenate(projected_point, axis=-1)
        elif point_type == "matrix":
            projected_point = [
                manifold_i.projection(point[..., i, :])
                for i, manifold_i in enumerate(self.manifolds)
            ]
            projected_point = gs.stack(projected_point, axis=-2)
        return projected_point

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        The tangent space of the product manifold is the direct sum of
        tangent spaces.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        point_type = self.default_point_type
        geomstats.errors.check_parameter_accepted_values(
            point_type, "point_type", ["vector", "matrix"]
        )

        if point_type == "vector":
            intrinsic = self.metric.is_intrinsic(base_point)
            tangent_vec = self._iterate_over_manifolds(
                "to_tangent", {"base_point": base_point, "vector": vector}, intrinsic
            )
            tangent_vec = gs.concatenate(tangent_vec, axis=-1)
        elif point_type == "matrix":
            tangent_vec = [
                manifold_i.to_tangent(vector[..., i, :], base_point[..., i, :])
                for i, manifold_i in enumerate(self.manifolds)
            ]
            tangent_vec = gs.stack(tangent_vec, axis=-2)
        return tangent_vec

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        The tangent space of the product manifold is the direct sum of
        tangent spaces.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        point_type = self.default_point_type
        geomstats.errors.check_parameter_accepted_values(
            point_type, "point_type", ["vector", "matrix"]
        )

        if point_type == "vector":
            intrinsic = self.metric.is_intrinsic(base_point)
            is_tangent = self._iterate_over_manifolds(
                "is_tangent",
                {"base_point": base_point, "vector": vector, "atol": atol},
                intrinsic,
            )
            is_tangent = gs.stack(is_tangent, axis=-1)

        else:
            is_tangent = gs.stack(
                [
                    space.is_tangent(
                        vector[..., i, :], base_point[..., i, :], atol=atol
                    )
                    for i, space in enumerate(self.manifolds)
                ],
                axis=-1,
            )

        is_tangent = gs.all(is_tangent, axis=-1)
        return is_tangent


class NFoldManifold(Manifold):
    r"""Class for an n-fold product manifold :math:`M^n`.

    Define a manifold as the product manifold of n copies of a given base manifold M.

    Parameters
    ----------
    base_manifold : Manifold
        Base manifold.
    n_copies : int
        Number of replication of the base manifold.
    metric : RiemannianMetric
        Metric object to use on the manifold.
    default_point_type : str, {\'vector\', \'matrix\'}
        Point type.
        Optional, default: 'vector'.
    default_coords_type : str, {\'intrinsic\', \'extrinsic\', etc}
        Coordinate type.
        Optional, default: 'intrinsic'.
    """

    def __init__(
        self,
        base_manifold,
        n_copies,
        metric=None,
        default_point_type="matrix",
        default_coords_type="intrinsic",
        **kwargs
    ):
        geomstats.errors.check_integer(n_copies, "n_copies")
        dim = n_copies * base_manifold.dim
        shape = (n_copies,) + base_manifold.shape

        super(NFoldManifold, self).__init__(
            dim=dim,
            shape=shape,
            default_point_type=default_point_type,
            default_coords_type=default_coords_type,
            **kwargs,
        )
        self.base_manifold = base_manifold
        self.base_shape = base_manifold.shape
        self.n_copies = n_copies
        if metric is None:
            self.metric = NFoldMetric(base_manifold.metric, n_copies)

    def belongs(self, point, atol=gs.atol):
        """Test if a point belongs to the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., n_copies, *base_shape]
            Point.
        atol : float,
            Tolerance.

        Returns
        -------
        belongs : array-like, shape=[..., n_copies, *base_shape]
            Boolean evaluating if the point belongs to the manifold.
        """
        point_ = gs.reshape(point, (-1, *self.base_shape))
        each_belongs = self.base_manifold.belongs(point_, atol=atol)
        reshaped = gs.reshape(each_belongs, (-1, self.n_copies))
        return gs.squeeze(gs.all(reshaped, axis=1))

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        The tangent space of the product manifold is the direct sum of
        tangent spaces.

        Parameters
        ----------
        vector : array-like, shape=[..., n_copies, *base_shape]
            Vector.
        base_point : array-like, shape=[..., n_copies, *base_shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        vector_, point_ = gs.broadcast_arrays(vector, base_point)
        point_ = gs.reshape(point_, (-1, *self.base_shape))
        vector_ = gs.reshape(vector_, (-1, *self.base_shape))
        each_tangent = self.base_manifold.is_tangent(vector_, point_)
        reshaped = gs.reshape(each_tangent, (-1, self.n_copies))
        return gs.all(reshaped, axis=1)

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        The tangent space of the product manifold is the direct sum of
        tangent spaces.

        Parameters
        ----------
        vector : array-like, shape=[..., n_copies, *base_shape]
            Vector.
        base_point : array-like, shape=[..., n_copies, *base_shape]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n_copies, *base_shape]
            Tangent vector at base point.
        """
        vector_, point_ = gs.broadcast_arrays(vector, base_point)
        point_ = gs.reshape(point_, (-1, *self.base_shape))
        vector_ = gs.reshape(vector_, (-1, *self.base_shape))
        each_tangent = self.base_manifold.to_tangent(vector_, point_)
        reshaped = gs.reshape(each_tangent, (-1, self.n_copies) + self.base_shape)
        return gs.squeeze(reshaped)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in the product space from the uniform distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples.
        bound : float
            Bound of the interval in which to sample for non compact manifolds.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n_copies, *base_shape]
            Points sampled on the product manifold.
        """
        sample = self.base_manifold.random_point(n_samples * self.n_copies, bound)
        reshaped = gs.reshape(sample, (n_samples, self.n_copies) + self.base_shape)
        return gs.squeeze(reshaped)

    def projection(self, point):
        """Project a point from product embedding manifold to the product manifold.

        Parameters
        ----------
        point : array-like, shape=[..., n_copies, *base_shape]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., n_copies, *base_shape]
            Projected point.
        """
        if hasattr(self.base_manifold, "projection"):
            point_ = gs.reshape(point, (-1, *self.base_shape))
            projected = self.base_manifold.projection(point_)
            reshaped = gs.reshape(projected, (-1, self.n_copies) + self.base_shape)
            return gs.squeeze(reshaped)
        raise NotImplementedError(
            "The base manifold does not implement a projection " "method."
        )


class NFoldMetric(RiemannianMetric):
    r"""Class for an n-fold product manifold :math:`M^n`.

    Define a manifold as the product manifold of n copies of a given base manifold M.

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
