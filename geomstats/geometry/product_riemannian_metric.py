"""Product of Riemannian metrics.

Define the metric of a product manifold endowed with a product metric.

Lead author: Nicolas Guigui, John Harvey.
"""

from math import prod

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.riemannian_metric import RiemannianMetric


def all_equal(arg):
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


class ProductRiemannianMetric(RiemannianMetric):
    """Class for product of Riemannian metrics.

    Parameters
    ----------
    metrics : list
        List of metrics in the product.
    scales : list
        List of positive values to rescale the inner product by on each factor.
        Note: To rescale the distances by a constant c, use c^2 for the scale
    default_point_type : str, {'vector', 'matrix'}
        Point type.
        Optional, default: 'vector'.
    """

    def __init__(self, metrics, default_point_type="vector", scales=None):
        geomstats.errors.check_parameter_accepted_values(
            default_point_type, "default_point_type", ["vector", "matrix"]
        )

        self.factors = metrics
        self._factor_dims = [factor.dim for factor in self.factors]
        self._factor_shapes = [factor.shape for factor in self.factors]
        self._factor_shape_sizes = [prod(metric.shape) for metric in self.factors]
        self._factor_signatures = [metric.signature for metric in self.factors]

        if scales is not None:
            for scale in scales:
                geomstats.errors.check_positive(scale, 'Each value in scales')
        self.scales = scales

        dim = sum(self._factor_dims)

        shape = self._find_product_shape(default_point_type)

        sig_pos = sum(sig[0] for sig in self._factor_signatures)
        sig_neg = sum(sig[1] for sig in self._factor_signatures)

        super().__init__(
            dim=dim,
            signature=(sig_pos, sig_neg),
            shape=shape
        )

        self.cum_index = gs.cumsum(self._factor_shape_sizes)[:-1]

    def _find_product_shape(self, default_point_type):
        """Determine an appropriate shape for the product from the factors."""
        if default_point_type == "vector":
            return (sum([prod(factor_shape) for factor_shape in self._factor_shapes]),)
        if not all_equal(self._factor_shapes):
            raise ValueError(
                "A default_point_type of \'matrix\' can only be used if all "
                "metrics have the same shape."
            )
        if not len(self._factor_shapes[0]) == 1:
            raise ValueError(
                "A default_point_type of \'matrix\' can only be used if all "
                "metrics have vector type."
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
            geomstats.errors.check_point_shape(point, factor)

        if self.default_point_type == 'vector':
            for response in points:
                start_of_coords = -1 * len(response.shape)
                if start_of_coords < -1:
                    response.reshape(response.shape[:start_of_coords] + (-1,))
            return gs.concatenate(points, axis=-1)
        else:
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
        geomstats.errors.check_point_shape(point, self)

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

    def _iterate_over_metrics(self, func, args):
        """Apply a function to each factor of the product.

        func is called on each factor of the product.

        Array-type arguments are separated out to be passed to func for each factor,
        but other arguments are passed unchanged.

        The results are returned in a list.

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
        out : list
            A list of the outputs from each factor, to be processed by the calling
            function.
        """
        # TODO The user may prefer to provide the arguments as lists and receive them as
        # TODO lists, as this may be the form in which they are available. This should
        # TODO be allowed, rather than packing and unpacking them repeatedly.

        args_list, numerical_args = \
            self._validate_and_prepare_args_for_iteration(args)

        out = [self._get_method(
            self.factors[i], func, args_list[i], numerical_args
        )
            for i in range(len(self.factors))]

        return out

    def _validate_and_prepare_args_for_iteration(self, args):
        """Separate arguments into different types and validate them.

        Parameters
        ----------
        args : dict
            Dict of arguments.
            Float or int arguments are passed to func for each factor
            Array-type arguments must be of type (..., shape)

        Returns
        -------
        arguments : list
            List of dicts of arguments with values being array-like.
            Each element of the list corresponds to a factor af the metric.
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
    def _get_method(metric, method_name, array_args, num_args):
        return getattr(metric, method_name)(**array_args, **num_args)

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
        factor_matrices = self._iterate_over_metrics(
            'metric_matrix', {'base_point': base_point})
        if self.scales is not None:
            factor_matrices = [
                matrix * scale for matrix, scale in zip(factor_matrices, self.scales)]
        metric_matrix = _block_diagonal(factor_matrices)
        return metric_matrix

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
        inner_products = self._iterate_over_metrics("inner_product", args)

        if self.scales is not None:
            inner_products = [
                product * scale for product, scale in zip(inner_products, self.scales)]

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
        exp = self._iterate_over_metrics("exp", args)

        if self.default_point_type == "vector":
            return gs.concatenate(exp, -1)
        return gs.stack(exp, axis=-2)

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
        logs = self._iterate_over_metrics("log", args)
        if self.default_point_type == "vector":
            return gs.concatenate(logs, axis=-1)
        return gs.stack(logs, axis=-2)


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
        super().__init__(dim=dim, shape=(n_copies, *base_shape))
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
