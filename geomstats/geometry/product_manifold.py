"""Product of manifolds.

Lead author: Nicolas Guigui, John Harvey.
"""


import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.product_riemannian_metric import (
    NFoldMetric,
    ProductRiemannianMetric,
    _all_equal,
    _IterateOverFactorsMixins,
)


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

    def __init__(self, factors, default_point_type="auto", **kwargs):
        if "metric_scales" in kwargs:
            raise TypeError(
                "Argument `metric_scales` is no longer in use: "
                "use `scale * metric` to achieved the desired behavior"
            )
        geomstats.errors.check_parameter_accepted_values(
            default_point_type,
            "default_point_type",
            ["auto", "vector", "matrix", "other"],
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

        kwargs.setdefault(
            "metric",
            ProductRiemannianMetric(
                [manifold.metric for manifold in factors],
                default_point_type=default_point_type,
            ),
        )

        super().__init__(
            pool_outputs=True,
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
            # TODO: need to revisit due to removal of scales
            self.embedding_space = ProductManifold(factor_embedding_spaces)

        self.cum_index = (
            gs.cumsum(self._factor_dims)[:-1]
            if self.default_coords_type == "intrinsic"
            else gs.cumsum(self.embedding_space._factor_dims)[:-1]
        )

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
    ):
        geomstats.errors.check_integer(n_copies, "n_copies")
        dim = n_copies * base_manifold.dim
        shape = (n_copies,) + base_manifold.shape

        if metric is None:
            metric = NFoldMetric(base_manifold.metric, n_copies)

        super().__init__(
            dim=dim,
            shape=shape,
            default_coords_type=base_manifold.default_coords_type,
            metric=metric,
        )

        self.base_manifold = base_manifold
        self.n_copies = n_copies

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
