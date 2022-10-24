"""Product of manifolds.

Lead author: Nicolas Guigui, John Harvey.
"""

from math import prod

import geomstats.backend as gs
import geomstats.errors
from geomstats.errors import ShapeError, check_point_shape
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.product_riemannian_metric import (
    NFoldMetric,
    ProductRiemannianMetric,
)


def broadcast_shapes(*args):
    """
    Broadcast the input shapes into a single shape.

    This is an adaptation of the version of the function implemented in mumpy 1.20.0

    Parameters
    ----------
    `*args` : tuples of ints, or ints
        The shapes to be broadcast against each other.

    Returns
    -------
    tuple
        Broadcasted shape.

    Raises
    ------
    ValueError
        If the shapes are not compatible and cannot be broadcast according
        to NumPy's broadcasting rules.
    """
    if len(args) == 0:
        return ()
    arrays = [gs.empty(x, dtype=[]) for x in args]
    broadcasted_array = gs.broadcast_arrays(*arrays)
    return broadcasted_array[0].shape


def all_equal(arg):
    """Check if all elements of arg are equal."""
    return arg.count(arg[0]) == len(arg)


class ProductManifold(Manifold):
    """Class for a product of manifolds M_1 x ... x M_n.

    In contrast to the classes NFoldManifold, Landmarks, or DiscretizedCurves,
    the manifolds M_1, ..., M_n need not be the same, nor of
    same dimension, but the list of manifolds needs to be provided.

    Parameters
    ----------
    factors : list
        List of manifolds in the product.
    metric_scales : list
        Optional. A list of positive numbers by which to scale the metric on each
        factor. If not given, no scaling is used.
    default_point_type : {'vector', 'matrix}
        Optional. Vector representation gives the point as a 1-d array.
        Matrix representation allows for a point to be represented by an array of shape
        (n, dim), if each manifold has default_point_type 'vector' with shape (dim,).
    """

    def __init__(
        self, factors, metric_scales=None, default_point_type="vector", **kwargs
    ):
        geomstats.errors.check_parameter_accepted_values(
            default_point_type, "default_point_type", ["vector", "matrix"]
        )

        self.factors = tuple(factors)
        self._factor_dims = [factor.dim for factor in self.factors]
        self._factor_shapes = [factor.shape for factor in self.factors]
        self._factor_default_coords_types = [
            factor.default_coords_type for factor in self.factors
        ]

        dim = sum(self._factor_dims)

        shape = self._find_product_shape(default_point_type)

        if "extrinsic" in self._factor_default_coords_types:
            default_coords_type = "extrinsic"
        else:
            default_coords_type = "intrinsic"

        if metric_scales is not None:
            for scale in metric_scales:
                geomstats.errors.check_positive(scale)
        kwargs.setdefault(
            "metric",
            ProductRiemannianMetric(
                [manifold.metric for manifold in factors],
                default_point_type=default_point_type,
                scales=metric_scales,
            ),
        )

        super().__init__(
            dim=dim,
            shape=shape,
            default_coords_type=default_coords_type,
            **kwargs,
        )

        if self.default_coords_type == "extrinsic":
            factor_embedding_spaces = [
                manifold.embedding_space
                if hasattr(manifold, "embedding_space")
                else manifold
                for manifold in factors
            ]
            self.embedding_space = ProductManifold(
                factor_embedding_spaces, metric_scales=metric_scales
            )

        self.cum_index = (
            gs.cumsum(self._factor_dims)[:-1]
            if self.default_coords_type == "intrinsic"
            else gs.cumsum(self.embedding_space._factor_dims)[:-1]
        )

    def _find_product_shape(self, default_point_type):
        """Determine an appropriate shape for the product from the factors."""
        if default_point_type == "vector":
            return (sum([prod(factor_shape) for factor_shape in self._factor_shapes]),)
        if not all_equal(self._factor_shapes):
            raise ValueError(
                "A default_point_type of 'matrix' can only be used if all "
                "manifolds have the same shape."
            )
        if not len(self._factor_shapes[0]) == 1:
            raise ValueError(
                "A default_point_type of 'matrix' can only be used if all "
                "manifolds have vector type."
            )
        return (len(self.factors), *self.factors[0].shape)

    def embed_to_product(self, points):
        """Map a point in each factor to a point in the product.

        Parameters
        ----------
        points : list
            A list of points, one from each factor, each array-like of shape
            (..., factor.shape)

        Returns
        -------
        point : array-like, shape (..., self.shape)

        Raises
        ------
        ShapeError
            If the points are not compatible with the shapes of the corresponding
            factors.
        """
        for point, factor in zip(points, self.factors):
            check_point_shape(point, factor)

        if self.default_point_type == "vector":
            return gs.concatenate(points, axis=-1)
        return gs.stack(points, axis=-2)

    def project_from_product(self, point):
        """Map a point in the product to points in each factor.

        Parameters
        ----------
        point : array-like, shape (..., self.shape)
            The point to be projected to the factors

        Returns
        -------
        projected_points : list of array-like
            The points on each factor, of shape (..., factor.shape)

        Raises
        ------
        ShapeError
            If the point does not have a shape compatible with the product manifold.
        """
        check_point_shape(point, self)

        if self.default_point_type == "vector":
            projected_points = gs.split(point, self.cum_index, axis=-1)
            projected_points = [
                self._reshape_trailing(projected_points[j], self.factors[j])
                for j in range(len(self.factors))
            ]

        else:
            projected_points = [point[..., j, :] for j in range(len(self.factors))]

        return projected_points

    @staticmethod
    def _reshape_trailing(argument, manifold):
        """Convert the trailing dimensions to match the shape of a factor manifold."""
        if manifold.default_coords_type == "vector":
            return argument
        leading_shape = argument.shape[:-1]
        trailing_shape = manifold.shape
        new_shape = leading_shape + trailing_shape
        return gs.reshape(argument, new_shape)

    def _iterate_over_factors(self, func, args):
        """Apply a function to each factor of the product.

        func is called on each factor of the product.

        Array-type arguments are separated out to be passed to func for each factor,
        but other arguments are passed unchanged.

        Parameters
        ----------
        func : str
            The name of a method which is defined for each factor of the product
            The method must return an array of shape (..., factor.shape) or a boolean
            array of shape (...,).
        args : dict
            Dict of arguments.
            Array-type arguments must be of type (..., shape)
            Other arguments are passed to each factor unchanged

        Returns
        -------
        out : array-like, shape = [..., {(), self.shape}]
        """
        # TODO The user may prefer to provide the arguments as lists and receive them as
        # TODO lists, as this may be the form in which they are available. This should
        # TODO be allowed, rather than packing and unpacking them repeatedly.
        args_list, numerical_args = self._validate_and_prepare_args_for_iteration(args)

        out = [
            self._get_method(self.factors[i], func, args_list[i], numerical_args)
            for i in range(len(self.factors))
        ]

        out = self._pool_outputs_from_function(out)
        return out

    def _validate_and_prepare_args_for_iteration(self, args):
        """Separate arguments into different types and validate them.

        Parameters
        ----------
        args : dict
            Dict of arguments.
            Float or int arguments are passed to func for each manifold
            Array-type arguments must be of type (..., shape)

        Returns
        -------
        arguments : list
            List of dicts of arguments with values being array-like.
            Each element of the list corresponds to a factor af the manifold.
        numerical_args : dict
            Dict of non-array arguments
        """
        args_list = [{} for _ in self.factors]
        numerical_args = {}
        for key, value in args.items():
            if not gs.is_array(value):
                numerical_args[key] = value
            else:
                new_args = self.project_from_product(value)
                for args_dict, new_arg in zip(args_list, new_args):
                    args_dict[key] = new_arg
        return args_list, numerical_args

    @staticmethod
    def _get_method(manifold, method_name, array_args, num_args):
        """Call manifold.method_name."""
        return getattr(manifold, method_name)(**array_args, **num_args)

    def _pool_outputs_from_function(self, outputs):
        """Collect outputs for each product to be returned.

        If each element of the output is a boolean array of the same shape, test along
        the list whether all elements are True and return a boolean array of the same
        shape.

        Otherwise, if each element of the output has a shape compatible with points of
        the corresponding factor, an attempt is made to map the list of points to a
        point in the product by embed_to_product.

        Parameters
        ----------
        outputs : list
            A list of outputs which must be pooled

        Returns
        -------
        pooled_output : array-like, shape {(...,), (..., self.shape)}
        """
        # TODO: simplify after cleaning gs.squeeze
        if (
            gs.all(
                [
                    gs.is_array(factor_output) or gs.is_bool(factor_output)
                    for factor_output in outputs
                ]
            )
            and all_equal([factor_output.shape for factor_output in outputs])
            and gs.all([gs.is_bool(factor_output) for factor_output in outputs])
        ):
            outputs = gs.stack(outputs)
            outputs = gs.all(outputs, axis=0)
            return outputs

        try:
            return self.embed_to_product(outputs)
        except ShapeError:
            raise RuntimeError(
                "Could not combine outputs - they are not points of the individual"
                " factors."
            )
        except ValueError:
            raise RuntimeError(
                "Could not combine outputs, probably because they could"
                " not be concatenated or stacked."
            )

    def belongs(self, point, atol=gs.atol):
        """Test if a point belongs to the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, embedding_space.dim,
            [n_manifolds, dim_each]}]
            Point.
        atol : float,
            Tolerance.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if the point belongs to the manifold.
        """
        belongs = self._iterate_over_factors("belongs", {"point": point, "atol": atol})
        return belongs

    def regularize(self, point):
        """Regularize the point into the manifold's canonical representation.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, embedding_space.dim,
            [n_manifolds, dim_each]}]
            Point to be regularized.

        Returns
        -------
        regularized_point : array-like, shape=[..., {dim, embedding_space.dim,
            [n_manifolds, dim_each]}]
            Point in the manifold's canonical representation.
        """
        regularized_point = self._iterate_over_factors("regularize", {"point": point})
        return regularized_point

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in the product space from the product distribution.

        The distribution used is the product of the distributions used by the
        random_sample methods of each individual factor manifold.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples.
        bound : float
            Bound of the interval in which to sample for non compact manifolds.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., {dim, embedding_space.dim,
            [n_manifolds, dim_each]}]
            Points sampled from the manifold.
        """
        samples = self._iterate_over_factors(
            "random_point", {"n_samples": n_samples, "bound": bound}
        )
        return samples

    def projection(self, point):
        """Project a point onto product manifold.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, embedding_space.dim,
            [n_manifolds, dim_each]}]
            Point in product manifold.

        Returns
        -------
        projected : array-like, shape=[..., {dim, embedding_space.dim,
            [n_manifolds, dim_each]}]
            Projected point.
        """
        projected_point = self._iterate_over_factors("projection", {"point": point})
        return projected_point

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

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

        Notes
        -----
        The tangent space of the product manifold is the direct sum of
        tangent spaces.
        """
        tangent_vec = self._iterate_over_factors(
            "to_tangent", {"base_point": base_point, "vector": vector}
        )
        return tangent_vec

    def is_tangent(self, vector, base_point=None, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        The tangent space of the product manifold is the direct sum of
        tangent spaces.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
            Optional, default: None
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        is_tangent = self._iterate_over_factors(
            "is_tangent", {"base_point": base_point, "vector": vector, "atol": atol}
        )
        return is_tangent


class NFoldManifold(Manifold):
    r"""Class for an n-fold product manifold :math:`M^n`.

    Define a manifold as the product manifold of n copies of a given base
    manifold M.

    Parameters
    ----------
    base_manifold : Manifold
        Base manifold.
    n_copies : int
        Number of replication of the base manifold.
    metric : RiemannianMetric
        Metric object to use on the manifold.
    default_coords_type : str, {\'intrinsic\', \'extrinsic\', etc}
        Coordinate type.
        Optional, default: 'intrinsic'.
    """

    def __init__(
        self,
        base_manifold,
        n_copies,
        metric=None,
        default_coords_type="intrinsic",
        **kwargs
    ):
        geomstats.errors.check_integer(n_copies, "n_copies")
        dim = n_copies * base_manifold.dim
        shape = (n_copies,) + base_manifold.shape

        super().__init__(
            dim=dim,
            shape=shape,
            default_coords_type=default_coords_type,
            **kwargs,
        )

        self.base_manifold = base_manifold
        self.n_copies = n_copies

        if metric is None:
            metric = NFoldMetric(base_manifold.metric, n_copies)
        self.metric = metric

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
        point_ = gs.reshape(point, (-1, *self.base_manifold.shape))
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
        point_ = gs.reshape(point_, (-1, *self.base_manifold.shape))
        vector_ = gs.reshape(vector_, (-1, *self.base_manifold.shape))
        each_tangent = self.base_manifold.is_tangent(vector_, point_, atol=atol)
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
        point_ = gs.reshape(point_, (-1, *self.base_manifold.shape))
        vector_ = gs.reshape(vector_, (-1, *self.base_manifold.shape))
        each_tangent = self.base_manifold.to_tangent(vector_, point_)
        reshaped = gs.reshape(
            each_tangent, (-1, self.n_copies) + self.base_manifold.shape
        )
        return gs.squeeze(reshaped)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in the product space from the product distribution.

        The distribution used is the product of the distributions that each copy of the
        manifold uses in its own random_point method.

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
        reshaped = gs.reshape(
            sample, (n_samples, self.n_copies) + self.base_manifold.shape
        )
        if n_samples > 1:
            return reshaped
        return gs.squeeze(reshaped, axis=0)

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
            point_ = gs.reshape(point, (-1, *self.base_manifold.shape))
            projected = self.base_manifold.projection(point_)
            reshaped = gs.reshape(
                projected, (-1, self.n_copies) + self.base_manifold.shape
            )
            return gs.squeeze(reshaped)
        raise NotImplementedError(
            "The base manifold does not implement a projection " "method."
        )
