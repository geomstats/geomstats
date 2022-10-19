"""Product of manifolds.

Lead author: Nicolas Guigui.
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
    metrics : list
        List of metrics for each manifold
        If a list of positive numbers is given, the metric is obtained from the manifold
        and scaled by the number.
        Optional, default value obtained from metric property of manifold
    default_point_type : {'vector', 'matrix}
        Optional. Vector representation gives the point as a 1-d array.
        Matrix representation allows for a point to be represented by an array of shape
        (n, dim), if each manifold has default_point_type 'vector' with shape (dim,).
    n_jobs : int
        Number of jobs for parallel computing.
        Optional, default: 1.
    """

    def __init__(self, factors, metrics=None, default_point_type="vector", **kwargs):
        geomstats.errors.check_parameter_accepted_values(
            default_point_type, "default_point_type", ["vector", "matrix"]
        )

        self.factors = factors

        dim = sum(self.factor_dims)

        if default_point_type == "vector":
            shape = (sum([prod(factor_shape) for factor_shape in self.factor_shapes]),)
        else:
            if (self.factor_shapes.count(self.factor_shapes[0]) ==
                    len(self.factor_shapes)):
                if len(self.factor_shapes[0]) == 1:
                    shape = (len(self.factors), *self.factors[0].shape)
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

        if "extrinsic" in self.factor_default_coords_types:
            default_coords_type = "extrinsic"
        else:
            default_coords_type = "intrinsic"

        if metrics is None:
            metric_scales = None
            metrics = [manifold.metric for manifold in factors]
        elif gs.all([is_positive(metric) for metric in metrics]):
            metric_scales = metrics
            metrics = [manifold.metric for manifold in factors]
        else:
            metric_scales = None

        kwargs.setdefault(
            "metric",
            ProductRiemannianMetric(
                metrics, default_point_type=default_point_type, scales=metric_scales),
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
            self.embedding_space = ProductManifold(factor_embedding_spaces)

    @property
    def factor_dims(self):
        """List containing the dimension of each factor."""
        return [factor.dim for factor in self.factors]

    @property
    def factor_shapes(self):
        """List containing the shape of each factor."""
        return [factor.shape for factor in self.factors]

    @property
    def factor_default_coords_types(self):
        """List containing the default_coords_type of each factor."""
        return [factor.default_coords_type for factor in self.factors]

    @staticmethod
    def _get_method(manifold, method_name, metric_args):
        return getattr(manifold, method_name)(**metric_args)

    def _validate_args_for_iteration(self, args):
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
            Shape of the leading dimensions of arguments, which must all match
        """
        arguments = {}
        numerical_args = {}
        leading_dimensions = []
        if self.default_point_type == "vector":
            cum_index = (
                gs.cumsum(self.factor_dims)[:-1]
                if self.default_coords_type == "intrinsic"
                else gs.cumsum(self.embedding_space.factor_dims)[:-1]
            )
            for key, value in args.items():
                if not gs.is_array(value):
                    numerical_args[key] = value
                else:
                    msg = (f"Argument {key}: {value} could not be broadcast to "
                           f"shape {self.factor_shapes}")
                    if value.shape[-1] != self.shape[-1]:
                        raise ValueError(msg)
                    leading_dimensions.append(value.shape[:-1])
                    arguments[key] = gs.split(value, cum_index, axis=-1)

        elif self.default_point_type == "matrix":
            for key, value in args.items():
                if not gs.is_array(value):
                    numerical_args[key] = value
                else:
                    arguments[key] = value
            for key, value in arguments.items():
                if value.shape[-2:] != self.shape[-2:]:
                    raise ValueError(
                        "Arguments did not have correct trailing dimensions"
                    )
                leading_dimensions.append(value.shape[:-2])

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
            self._validate_args_for_iteration(args)

        if self.default_point_type == "vector":
            args_list = [
                {
                    key: self._reshape_trailing(arguments[key][j], self.factors[j])
                    for key in arguments
                }
                for j in range(len(self.factors))
            ]
        elif self.default_point_type == "matrix":
            args_list = [
                {key: arguments[key][..., j, :] for key in arguments}
                for j in range(len(self.factors))
            ]

        out = [self._get_method(
            self.factors[i], func, {**args_list[i], **numerical_args}
        )
            for i in range(len(self.factors))]

        if self.default_point_type == 'vector':
            # The individual factors might have matrix type, so their responses must be
            # flattened before proceeding. Since it is not straightforward to see which
            # type has been returned, we are going to have to compare the shape of the
            # return to the leading shape of all arguments broadcast against each otehr
            for response in out:
                if (gs.is_array(response) and
                        len(response.shape) > len(leading_dimensions) + 1):
                    response.reshape(response.shape[:-2] + (-1,))
            out = gs.concatenate(out, axis=-1)
        else:
            out = gs.stack(out, axis=-2)
        return out

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

        Each factor has a method random_sample which sets the distribution for that
        factor.

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
