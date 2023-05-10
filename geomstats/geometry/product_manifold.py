"""Product of manifolds.

Lead author: Nicolas Guigui, John Harvey.
"""
import math

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.complex_manifold import ComplexManifold
from geomstats.geometry.complex_riemannian_metric import ComplexRiemannianMetric
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric

COMPLEX_OBJECTS = (ComplexRiemannianMetric, ComplexManifold)


def _factor_is_complex(factor):
    if (
        isinstance(factor, COMPLEX_OBJECTS)
        or hasattr(factor, "underlying_metric")
        and isinstance(factor.underlying_metric, COMPLEX_OBJECTS)
    ):
        return True

    return False


def _has_mixed_fields(factors):
    bools = [_factor_is_complex(factor) for factor in factors]
    if len(set(bools)) == 2:
        return True

    return False


def _all_equal(arg):
    """Check if all elements of arg are equal."""
    return arg.count(arg[0]) == len(arg)


def _block_diagonal(factor_matrices):
    """Put a list of square matrices in block diagonal form."""
    shapes_dict = {}
    for i, matrix_i in enumerate(factor_matrices):
        for j, matrix_j in enumerate(factor_matrices):
            shapes_dict[(i, j)] = matrix_i.shape[:-1] + matrix_j.shape[-1:]
    rows = []
    # concacatenate along axis = -2
    for (i, matrix_i) in enumerate(factor_matrices):
        # concatenate along axis = -1
        blocks_to_concatenate = []
        for j, _ in enumerate(factor_matrices):
            if i == j:
                blocks_to_concatenate.append(matrix_i)
            else:
                blocks_to_concatenate.append(gs.zeros(shapes_dict[(i, j)]))
        row = gs.concatenate(blocks_to_concatenate, axis=-1)
        rows.append(row)
    metric_matrix = gs.concatenate(rows, axis=-2)
    return metric_matrix


def _find_product_shape(factors, default_point_type):
    """Determine an appropriate shape for the product from the factors."""
    factor_shapes = [factor.shape for factor in factors]

    if default_point_type == "auto":
        if _all_equal(factor_shapes):
            return len(factors), *factors[0].shape
        default_point_type = "vector"
    if default_point_type == "vector":
        return (sum(math.prod(factor_shape) for factor_shape in factor_shapes),)
    if not _all_equal(factor_shapes):
        raise ValueError(
            "A default_point_type of 'matrix' or 'other' can only be used if all "
            "manifolds have the same shape."
        )
    if default_point_type == "matrix" and not len(factor_shapes[0]) == 1:
        raise ValueError(
            "A default_point_type of 'matrix' can only be used if all "
            "manifolds have vector type."
        )
    return len(factors), *factors[0].shape


class _IterateOverFactorsMixins:
    def __init__(
        self, factors, cum_index, pool_outputs, has_mixed_fields, *args, **kwargs
    ):
        self.factors = factors
        self._cum_index = cum_index
        self._pool_outputs = pool_outputs
        self._has_mixed_fields = has_mixed_fields
        super().__init__(*args, **kwargs)

    def embed_to_product(self, points):
        """Map a point in each factor to a point in the product.

        Parameters
        ----------
        points : list
            A list of points, one from each factor, each array-like of shape
            (..., factor.shape)

        Returns
        -------
        point : array-like, shape (..., shape)

        Raises
        ------
        ShapeError
            If the points are not compatible with the shapes of the corresponding
            factors.
        """
        for point, factor in zip(points, self.factors):
            geomstats.errors.check_point_shape(point, factor)

        if self.default_point_type == "vector":
            points_ = []
            for response, factor in zip(points, self.factors):
                if gs.ndim(response) > len(factor.shape):
                    response = gs.reshape(response, (-1, math.prod(response.shape[1:])))
                else:
                    response = gs.flatten(response)

                points_.append(response)
            return gs.concatenate(points_, axis=-1)
        stacking_axis = -1 * len(self.shape)
        return gs.stack(points, axis=stacking_axis)

    def project_from_product(self, point):
        """Map a point in the product to points in each factor.

        Parameters
        ----------
        point : array-like, shape (..., shape)
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
        geomstats.errors.check_point_shape(point, self)

        if self.default_point_type == "vector":
            projected_points = gs.split(point, self._cum_index, axis=-1)
            projected_points = [
                self._reshape_trailing(projected_points[j], self.factors[j])
                for j in range(len(self.factors))
            ]

        else:
            splitting_axis = -1 * len(self.shape)
            projected_points = gs.split(point, len(self.factors), axis=splitting_axis)
            projected_points = [
                gs.squeeze(projected_point, axis=splitting_axis)
                for projected_point in projected_points
            ]

        if self._has_mixed_fields:
            for i, (factor, projected_point) in enumerate(
                zip(self.factors, projected_points)
            ):
                if not _factor_is_complex(factor):
                    projected_points[i] = gs.real(projected_point)

        return projected_points

    @staticmethod
    def _reshape_trailing(argument, factor):
        """Convert the trailing dimensions to match the shape of a factor manifold."""
        space = factor._space if isinstance(factor, RiemannianMetric) else factor

        if space.default_coords_type == "vector":
            return argument
        leading_shape = argument.shape[:-1]
        trailing_shape = space.shape
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
        out : array-like, shape = [..., {(), shape}]
        """
        # TODO The user may prefer to provide the arguments as lists and receive them as
        # TODO lists, as this may be the form in which they are available. This should
        # TODO be allowed, rather than packing and unpacking them repeatedly.
        args_list, numerical_args = self._validate_and_prepare_args_for_iteration(args)

        out = [
            self._get_method(self.factors[i], func, args_list[i], numerical_args)
            for i in range(len(self.factors))
        ]
        if self._pool_outputs:
            return self._pool_outputs_from_function(out)
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
    def _get_method(factor, method_name, array_args, num_args):
        """Call factor.method_name."""
        return getattr(factor, method_name)(**array_args, **num_args)


class ProductManifold(_IterateOverFactorsMixins, Manifold):
    """Class for a product of manifolds M_1 x ... x M_n.

    In contrast to the classes NFoldManifold, Landmarks, or DiscretizedCurves,
    the manifolds M_1, ..., M_n need not be the same, nor of
    same dimension, but the list of manifolds needs to be provided.

    Parameters
    ----------
    factors : list
        List of manifolds in the product.
    default_point_type : {'auto', 'vector', 'matrix', 'other'}
        Optional. Default value is 'auto', which will implement as 'vector' unless all
        factors have the same shape. Vector representation gives the point as a 1-d
        array. Matrix representation allows for a point to be represented by an array of
        shape (n, dim), if each manifold has default_point_type 'vector' with shape
        (dim,). 'other' will behave as `matrix` but for higher dimensions.
    """

    def __init__(self, factors, default_point_type="auto", equip=True):
        geomstats.errors.check_parameter_accepted_values(
            default_point_type,
            "default_point_type",
            ["auto", "vector", "matrix", "other"],
        )

        factors = tuple(factors)

        factor_dims = [factor.dim for factor in factors]
        factor_default_coords_types = [factor.default_coords_type for factor in factors]

        dim = sum(factor_dims)

        shape = _find_product_shape(factors, default_point_type)

        if "extrinsic" in factor_default_coords_types:
            default_coords_type = "extrinsic"
        else:
            default_coords_type = "intrinsic"

        if default_coords_type == "extrinsic":
            factor_embedding_spaces = [
                manifold.embedding_space
                if hasattr(manifold, "embedding_space")
                else manifold
                for manifold in factors
            ]
            # TODO: need to revisit due to removal of scales
            self.embedding_space = ProductManifold(factor_embedding_spaces)

        cum_index = (
            gs.cumsum(factor_dims)[:-1]
            if default_coords_type == "intrinsic"
            else self.embedding_space._cum_index
        )

        super().__init__(
            factors=factors,
            cum_index=cum_index,
            pool_outputs=True,
            has_mixed_fields=_has_mixed_fields(factors),
            dim=dim,
            shape=shape,
            default_coords_type=default_coords_type,
            equip=equip,
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return ProductRiemannianMetric

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
        all_arrays = gs.all([gs.is_array(factor_output) for factor_output in outputs])
        if (
            all_arrays
            and _all_equal([factor_output.shape for factor_output in outputs])
            and gs.all([gs.is_bool(factor_output) for factor_output in outputs])
            or (not all_arrays)
        ):
            outputs = gs.stack([gs.array(factor_output) for factor_output in outputs])
            outputs = gs.all(outputs, axis=0)
            return outputs

        try:
            return self.embed_to_product(outputs)
        except geomstats.errors.ShapeError:
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
        random_point methods of each individual factor manifold.

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

    def random_tangent_vec(self, base_point, n_samples=1):
        """Sample on the tangent space from the product distribution.

        The distribution used is the product of the distributions used by the
        random_tangent_vec methods of each individual factor manifold.

        Parameters
        ----------
        base_point : array-like, shape=[..., n, n]
            Base point of the tangent space.
            Optional, default: None.
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., {dim, embedding_space.dim,
            [n_manifolds, dim_each]}]
            Points sampled in the tangent space of the product manifold at base_point.
        """
        samples = self._iterate_over_factors(
            "random_tangent_vec", {"base_point": base_point, "n_samples": n_samples}
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


class ProductRiemannianMetric(_IterateOverFactorsMixins, RiemannianMetric):
    """Class for product of Riemannian metrics."""

    def __init__(self, space):
        factors = [factor.metric for factor in space.factors]
        factor_signatures = [metric.signature for metric in factors]

        sig_pos = sum(sig[0] for sig in factor_signatures)
        sig_neg = sum(sig[1] for sig in factor_signatures)

        super().__init__(
            space=space,
            factors=factors,
            cum_index=space._cum_index,
            pool_outputs=False,
            has_mixed_fields=space._has_mixed_fields,
            signature=(sig_pos, sig_neg),
        )

    @property
    def shape(self):
        """Shape of space."""
        return self._space.shape

    @property
    def default_point_type(self):
        """Point type of space."""
        return self._space.default_point_type

    def metric_matrix(self, base_point=None):
        """Compute the matrix of the inner-product.

        Matrix of the inner-product defined by the Riemmanian metric
        at point base_point of the manifold.

        Parameters
        ----------
        base_point : array-like, shape=[..., self.shape]
            Point on the manifold at which to compute the inner-product matrix.
            Optional, default: None.

        Returns
        -------
        matrix : array-like, shape as described below
            Matrix of the inner-product at the base point.
            The matrix is in block diagonal form with a block for each factor.
            Each block is the same size as the metric_matrix for that factor.
        """
        factor_matrices = self._iterate_over_factors(
            "metric_matrix", {"base_point": base_point}
        )
        return _block_diagonal(factor_matrices)

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
        tangent_vec_a : array-like, shape=[..., self.shape]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., self.shape]
            Second tangent vector at base point.
        base_point : array-like, shape=[..., self.shape]
            Point on the manifold.
            Optional, default: None.

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Inner-product of the two tangent vectors.
        """
        args = {
            "tangent_vec_a": tangent_vec_a,
            "tangent_vec_b": tangent_vec_b,
            "base_point": base_point,
        }
        inner_products = self._iterate_over_factors("inner_product", args)
        return sum(inner_products)

    def exp(self, tangent_vec, base_point=None, **kwargs):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., self.shape]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., self.shape]
            Point on the manifold.
            Optional, default: None.

        Returns
        -------
        exp : array-like, shape=[..., self.shape]
            Point on the manifold equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        args = {"tangent_vec": tangent_vec, "base_point": base_point}
        exp = self._iterate_over_factors("exp", args)

        if self.default_point_type == "vector":
            return gs.concatenate(exp, -1)
        return gs.stack(exp, axis=-len(self.shape))

    def log(self, point, base_point=None, **kwargs):
        """Compute the Riemannian logarithm of a point.

        Parameters
        ----------
        point : array-like, shape=[..., self.shape]
            Point on the manifold.
        base_point : array-like, shape=[..., self.shape]
            Point on the manifold.
            Optional, default: None.

        Returns
        -------
        log : array-like, shape=[..., self.shape]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        args = {"point": point, "base_point": base_point}
        logs = self._iterate_over_factors("log", args)
        if self.default_point_type == "vector":
            return gs.concatenate(logs, axis=-1)
        return gs.stack(logs, axis=-len(self.shape))
