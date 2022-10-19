"""Product of manifolds.

Lead author: Nicolas Guigui, John Harvey.
"""

from math import prod

import geomstats.backend as gs
import geomstats.errors
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


def is_positive(arg):
    """Check if arg is a positive number."""
    if isinstance(arg, (int, float)):
        return arg > 0
    return False


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
    n_jobs : int
        Number of jobs for parallel computing.
        Optional, default: 1.
    """

    def __init__(
            self, factors, metric_scales=None, default_point_type="vector", **kwargs):
        geomstats.errors.check_parameter_accepted_values(
            default_point_type, "default_point_type", ["vector", "matrix"]
        )

        self.factors = tuple(factors)
        self._factor_dims = [factor.dim for factor in self.factors]
        self._factor_shapes = [factor.shape for factor in self.factors]
        self._factor_default_coords_types = [
            factor.default_coords_type for factor in self.factors]

        dim = sum(self._factor_dims)

        shape = self._find_product_shape(default_point_type)

        if "extrinsic" in self._factor_default_coords_types:
            default_coords_type = "extrinsic"
        else:
            default_coords_type = "intrinsic"

        if metric_scales is not None:
            for scale in metric_scales:
                geomstats.errors.check_positive(scale)
        kwargs["metric"] = ProductRiemannianMetric(
            [manifold.metric for manifold in factors],
            default_point_type=default_point_type,
            scales=metric_scales
        )

        super().__init__(
            dim=dim,
            shape=shape,
            default_coords_type=default_coords_type,
            **kwargs,
        )

        if self.default_coords_type == "extrinsic":
            factor_embedding_spaces = [
                manifold.embedding_space if manifold.default_coords_type == "extrinsic"
                else manifold
                for manifold in factors
            ]
            self.embedding_space = ProductManifold(
                factor_embedding_spaces, metric_scales=metric_scales)

        self.cum_index = (
            gs.cumsum(self._factor_dims)[:-1]
            if self.default_coords_type == "intrinsic"
            else gs.cumsum(self.embedding_space._factor_dims)[:-1]
        )

    def _find_product_shape(self, default_point_type):
        if default_point_type == "vector":
            return (sum([prod(factor_shape) for factor_shape in self._factor_shapes]),)
        else:
            if (self._factor_shapes.count(self._factor_shapes[0]) ==
                    len(self._factor_shapes)):
                if len(self._factor_shapes[0]) == 1:
                    return (len(self.factors), *self.factors[0].shape)
                else:
                    raise ValueError(
                        "A default_point_type of \'matrix\' can only be used if all "
                        "manifolds have vector type."
                    )
            else:
                raise ValueError(
                    "A default_point_type of \'matrix\' can only be used if all "
                    "manifolds have the same shape."
                )

    def embed_to_product(self, points, leading_dimensions):
        """Map a point in each factor to a point in the product."""
        # TODO Seems somewhat more complicated than necessary
        if self.default_point_type == 'vector':
            # The individual factors might have matrix type, so their responses must be
            # flattened before proceeding. Since it is not straightforward to see which
            # type has been returned, we are going to have to compare the shape of the
            # return to the leading shape of all arguments broadcast against each otehr
            for response in points:
                if (gs.is_array(response) and
                        len(response.shape) > len(leading_dimensions) + 1):
                    response.reshape(response.shape[:-2] + (-1,))
            return gs.concatenate(points, axis=-1)
        else:
            return gs.stack(points, axis=-2)

    def project_from_product(self, point, include_leading_dimensions=False):
        """Map a point in the product to points in each factor.

        Can also return the shape of the leading_dimensions accompanying point
        """
        leading_dimensions = []
        shape_error_msg = (f"The shape of {point}, which is {point.shape} is not"
                           f" compatible with the shape of the manifold, {self.shape}")

        if self.default_point_type == "vector":
            if point.shape[-1] != self.shape[-1]:
                raise ValueError(shape_error_msg)
            leading_dimensions.append(point.shape[:-1])
            projected_points = gs.split(point, self.cum_index, axis=-1)

        elif self.default_point_type == "matrix":
            if point.shape[-2:] != self.shape[-2:]:
                raise ValueError(shape_error_msg)
            leading_dimensions.append(point.shape[:-2])
            projected_points = [point[..., j, :] for j in range(len(self.factors))]

        leading_dimensions = broadcast_shapes(*leading_dimensions)

        if include_leading_dimensions:
            return projected_points, leading_dimensions
        return projected_points

    def _iterate_over_factors(self, func, args):
        """Apply a function to each factor of the product.

        If default_point_type is 'vector' then the vector must be split up into
        sub-vectors for each component, and then those sub-vectors should be broadcast
        into the correct shape for the individual manifold. These are then passed to the
        function along with any non-array arguments.

        However, if default_point_type is 'matrix', then we have an array of shape
        (..., n_manifolds, dim_each). In this case we simply split into n_manifolds
        arrays, each of shape (dim_each,).

        The returned value will be array-like with all outputs contained in the
        trailing dimension if default_point_type is vector, or in the two trailing
        dimensions if default_point_type is matrix.

        Parameters
        ----------
        func : str
            The name of a method which is defined for each factor of the product
            The method returns an array of shape (..., k)
        args : dict
            Dict of arguments.
            Array-type arguments must be of type (..., shape)
            Other arguments are passed to each factor unchanged

        Returns
        -------
        out : array-like, shape = [..., {n_manifolds*k, (n_manifolds, k)}]
        """
        arguments, numerical_args, leading_dimensions = \
            self._validate_and_prepare_args_for_iteration(args)

        args_list = [
            {key: arguments[key][j] for key in arguments}
            for j in range(len(self.factors))
        ]

        out = [self._get_method(
            self.factors[i], func, args_list[i], numerical_args
        )
            for i in range(len(self.factors))]

        out = self._pool_outputs_from_function(out, leading_dimensions)
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
        arguments : dict
            Dict of arguments with values being lists of array-like arguments.
            Each array-like argument corresponds to a factor af the manifold.
            The trailing dimensions of the argument match the shape of the factor
        numerical_args : dict
            Dict of non-array arguments
        leading_dimensions : tuple
            Shape of the leading dimensions of arguments, which must be broadcastable
        """
        arguments = {}
        numerical_args = {}
        leading_dimensions = []
        for key, value in args.items():
            if not gs.is_array(value):
                numerical_args[key] = value
            else:
                args_to_append, dim_to_append = self.project_from_product(
                    value, include_leading_dimensions=True)
                if self.default_point_type == "vector":
                    arguments[key] = [
                        self._reshape_trailing(args_to_append[j], self.factors[j])
                        for j in range(len(self.factors))
                    ]
                arguments[key] = args_to_append
                leading_dimensions.append(dim_to_append)

        leading_dimensions = broadcast_shapes(*leading_dimensions)
        return arguments, numerical_args, leading_dimensions

    @staticmethod
    def _reshape_trailing(argument, manifold):
        """Convert the trailing dimensions to match the shape of a factor manifold."""
        if manifold.default_coords_type == "vector":
            return argument
        leading_shape = argument.shape[:-1]
        trailing_shape = manifold.shape
        new_shape = leading_shape + trailing_shape
        return gs.reshape(argument, new_shape)

    @staticmethod
    def _get_method(manifold, method_name, array_args, num_args):
        return getattr(manifold, method_name)(**array_args, **num_args)

    def _pool_outputs_from_function(self, out, leading_dimensions):
        """Collect outputs for each product to be returned.

        At the moment, this only replicates the existing, non-functional behaviour.
        This is caused by assuming that everything more or less looks like a point.
        # TODO Rewrite

        Each element of the output has some shape. If we strip off the
        leading dimensions we can see the shape of the output
        If it is empty, we can assume the output is of boolean type or else a simple
        number
        If the output is a boolean, we will want to take AND along some particular axis.
        If the output is a number, we probably want to add them all up. The function
        which calls _iterate_over_factors should probably tell us
        If the output is of shape point, we will want to apply embed_to_product
        """
        return self.embed_to_product(out, leading_dimensions)

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
        belongs = self._iterate_over_factors(
            "belongs", {"point": point, "atol": atol}
        )
        belongs = gs.all(belongs, axis=-1)
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
        regularized_point = self._iterate_over_factors(
            "regularize", {"point": point}
        )
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
        is_tangent = self._iterate_over_factors(
            "is_tangent", {"base_point": base_point, "vector": vector, "atol": atol}
        )
        is_tangent = gs.all(is_tangent, axis=-1)
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
        self.base_shape = base_manifold.shape
        self.n_copies = n_copies

        self.metric = metric
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
            point_ = gs.reshape(point, (-1, *self.base_shape))
            projected = self.base_manifold.projection(point_)
            reshaped = gs.reshape(projected, (-1, self.n_copies) + self.base_shape)
            return gs.squeeze(reshaped)
        raise NotImplementedError(
            "The base manifold does not implement a projection " "method."
        )
