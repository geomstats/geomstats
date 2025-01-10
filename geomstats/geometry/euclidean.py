"""Euclidean space."""

import math

import geomstats.backend as gs
from geomstats.geometry.base import VectorSpace
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.vectorization import check_is_batch, repeat_out


class Euclidean(VectorSpace):
    """Class for Euclidean spaces.

    By definition, a Euclidean space is a vector space of a given
    dimension, equipped with a Euclidean metric.

    Parameters
    ----------
    dim : int
        Dimension of the Euclidean space.
    """

    def __init__(self, dim, equip=True):
        super().__init__(
            dim=dim,
            shape=(dim,),
            equip=equip,
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return CanonicalEuclideanMetric

    @property
    def identity(self):
        """Identity of the group.

        Returns
        -------
        identity : array-like, shape=[n]
        """
        return gs.zeros(self.dim)

    def _create_basis(self):
        """Create the canonical basis."""
        return gs.eye(self.dim)

    def exp(self, tangent_vec, base_point):
        """Compute the group exponential, which is simply the addition.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n]
            Point from which the exponential is computed.

        Returns
        -------
        point : array-like, shape=[..., n]
            Group exponential.
        """
        return tangent_vec + base_point


class EuclideanMetric(RiemannianMetric):
    """Class for a Euclidean metric.

    This metric is:

    - flat: the inner-product is independent of the base point;
    - positive definite
    """

    def __init__(self, space, metric_matrix=None, signature=None):
        super().__init__(space, signature=signature)
        self._check_metric_matrix_dim(space, metric_matrix)
        if metric_matrix is None and space.point_ndim == 1:
            metric_matrix = gs.eye(space.dim)

        self.metric_matrix_ = metric_matrix

    @staticmethod
    def _check_metric_matrix_dim(space, metric_matrix):
        """Check metric matrix dimension."""
        if metric_matrix is None:
            return

        expected_shape = (space.dim, space.dim)
        if metric_matrix.shape != expected_shape:
            raise ValueError(
                f"metric_matrix shape is {metric_matrix.shape};"
                f"expected: {expected_shape}"
            )

    def metric_matrix(self, base_point=None):
        """Compute the inner-product matrix, independent of the base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        inner_prod_mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        if self._space.point_ndim > 1:
            raise NotImplementedError("`metric_matrix` is not implemented.")

        dim = self._space.dim
        return repeat_out(
            self._space.point_ndim,
            gs.copy(self.metric_matrix_),
            base_point,
            out_shape=(dim, dim),
        )

    def inner_product_derivative_matrix(self, base_point=None):
        r"""Compute derivative of the inner prod matrix at base point.

        Writing :math:`g_{ij}` the inner-product matrix at base point,
        this computes :math:`mat_{ijk} = \partial_k g_{ij}`, where the
        index k of the derivation is put last.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        metric_derivative : array-like, shape=[..., dim, dim, dim]
            Derivative of the inner-product matrix, where the index
            k of the derivation is last: :math:`mat_{ijk} = \partial_k g_{ij}`.
        """
        if self._space.point_ndim > 1:
            raise NotImplementedError(
                "`inner_product_derivative_matrix` is not implemented."
            )

        dim = self._space.dim
        shape = (dim, dim, dim)
        return repeat_out(
            self._space.point_ndim, gs.zeros(shape), base_point, out_shape=shape
        )

    def christoffels(self, base_point=None):
        """Christoffel symbols associated with the connection.

        The contravariant index is on the first dimension.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        gamma : array-like, shape=[..., dim, dim, dim]
            Christoffel symbols, with the contravariant index on
            the first dimension.
        """
        if self._space.point_ndim > 1:
            raise NotImplementedError("The Christoffel symbols are not implemented.")

        dim = self._space.dim
        shape = (dim, dim, dim)
        gamma = gs.zeros(shape)
        return repeat_out(self._space.point_ndim, gamma, base_point, out_shape=shape)

    def exp(self, tangent_vec, base_point):
        """Compute exp map of a base point in tangent vector direction.

        The Riemannian exponential is vector addition in the Euclidean space.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            Riemannian exponential.
        """
        return base_point + tangent_vec

    def log(self, point, base_point):
        """Compute log map using a base point and other point.

        The Riemannian logarithm is the subtraction in the Euclidean space.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point.
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        log : array-like, shape=[..., dim]
            Riemannian logarithm.
        """
        return point - base_point

    def parallel_transport(
        self, tangent_vec, base_point=None, direction=None, end_point=None
    ):
        r"""Compute the parallel transport of a tangent vector.

        On a Euclidean space, the parallel transport of a (tangent) vector
        returns the vector itself.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., dim]
            Point on the manifold. Point to transport from.
            Optional, default: None
        direction : array-like, shape=[..., dim]
            Tangent vector at base point, along which the parallel transport
            is computed.
            Optional, default: None.
        end_point : array-like, shape=[..., dim]
            Point on the manifold. Point to transport to.
            Optional, default: None.

        Returns
        -------
        transported_tangent_vec : array-like, shape=[..., dim]
            Transported tangent vector at `exp_(base_point)(tangent_vec_b)`.
        """
        transported_tangent_vec = gs.copy(tangent_vec)
        return repeat_out(
            self._space.point_ndim,
            transported_tangent_vec,
            tangent_vec,
            base_point,
            direction,
            end_point,
            out_shape=self._space.shape,
        )

    def geodesic(self, initial_point, end_point=None, initial_tangent_vec=None):
        """Generate parameterized function for the geodesic curve.

        Geodesic curve defined by either:

        - an initial point and an initial tangent vector,
        - an initial point and an end point.

        Parameters
        ----------
        initial_point : array-like, shape=[..., dim]
            Point on the manifold, initial point of the geodesic.
        end_point : array-like, shape=[..., dim], optional
            Point on the manifold, end point of the geodesic. If None,
            an initial tangent vector must be given.
        initial_tangent_vec : array-like, shape=[..., dim], optional
            Tangent vector at base point, the initial speed of the geodesics.
            Optional, default: None.
            If None, an end point must be given and a logarithm is computed.

        Returns
        -------
        path : callable
            Time parameterized geodesic curve. If a batch of initial
            conditions is passed, the output array's first dimension
            represents the different initial conditions, and the second
            corresponds to time.
        """
        if end_point is None and initial_tangent_vec is None:
            raise ValueError(
                "Specify an end point or an initial tangent "
                "vector to define the geodesic."
            )
        if end_point is not None:
            if initial_tangent_vec is not None:
                raise ValueError(
                    "Cannot specify both an end point and an initial tangent vector."
                )

            initial_tangent_vec = self.log(end_point, initial_point)

        is_batch = check_is_batch(
            self._space.point_ndim, initial_point, initial_tangent_vec
        )
        if is_batch:
            initial_point = gs.expand_dims(
                initial_point, axis=-(self._space.point_ndim + 1)
            )

        ijk = "ijk"[: self._space.point_ndim]

        def path(t):
            """Generate parameterized function for geodesic curve.

            Parameters
            ----------
            t : array-like, shape=[n_points,]
                Times at which to compute points of the geodesics.
            """
            t = gs.array(t)
            t = gs.cast(t, initial_tangent_vec.dtype)
            t = gs.to_ndarray(t, to_ndim=1)
            tangent_vecs = gs.einsum(f"n,...{ijk}->...n{ijk}", t, initial_tangent_vec)
            return initial_point + tangent_vecs

        return path

    def injectivity_radius(self, base_point=None):
        """Compute the radius of the injectivity domain.

        This is is the supremum of radii r for which the exponential map is a
        diffeomorphism from the open ball of radius r centered at the base
        point onto its image.

        Parameters
        ----------
        base_point : array-like, shape=[..., {dim, [n, m]}]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        radius = gs.array(math.inf)
        return repeat_out(self._space.point_ndim, radius, base_point)


class CanonicalEuclideanMetric(EuclideanMetric):
    """Class for the canonical Euclidean metric.

    Notes
    -----
    Metric matrix is identity (NB: `EuclideanMetric` allows
    to use a different metric matrix).
    """

    def __init__(self, space):
        super().__init__(space)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Inner product between two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a: array-like, shape=[..., dim]
            Tangent vector at base point.
        tangent_vec_b: array-like, shape=[..., dim]
            Tangent vector at base point.
        base_point: array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """
        inner_product = gs.dot(tangent_vec_a, tangent_vec_b)
        return repeat_out(
            self._space.point_ndim,
            inner_product,
            tangent_vec_a,
            tangent_vec_b,
            base_point,
        )

    def squared_norm(self, vector, base_point=None):
        """Compute the square of the norm of a vector.

        Squared norm of a vector associated to the inner product
        at the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        sq_norm : array-like, shape=[...,]
            Squared norm.
        """
        sq_norm = gs.linalg.norm(vector, axis=-1) ** 2
        return repeat_out(self._space.point_ndim, sq_norm, vector, base_point)

    def norm(self, vector, base_point=None):
        """Compute norm of a vector.

        Norm of a vector associated to the inner product
        at the tangent space at a base point.

        Note: This only works for positive-definite
        Riemannian metrics and inner products.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        norm : array-like, shape=[...,]
            Norm.
        """
        norm = gs.linalg.norm(vector, axis=-1)
        return repeat_out(self._space.point_ndim, norm, vector, base_point)
