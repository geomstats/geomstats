"""Manifold module.

In other words, a topological space that locally resembles
Euclidean space near each point.

Lead author: Nina Miolane.
"""

import abc

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.riemannian_metric import RiemannianMetric


class Manifold(abc.ABC):
    r"""Class for manifolds.

    Parameters
    ----------
    dim : int
        Dimension of the manifold.
    shape : tuple of int
        Shape of one element of the manifold.
        Optional, default : None.
    metric : RiemannianMetric
        Metric object to use on the manifold.
    default_coords_type : str, {\'intrinsic\', \'extrinsic\', etc}
        Coordinate type.
        Optional, default: 'intrinsic'.
    """

    def __init__(
        self, dim, shape, metric=None, default_coords_type="intrinsic", **kwargs
    ):
        super().__init__(**kwargs)
        geomstats.errors.check_integer(dim, "dim")

        if not isinstance(shape, tuple):
            raise ValueError("Expected a tuple for the shape argument.")

        self.dim = dim
        self.shape = shape
        self.shape_dim = gs.prod(shape)
        self.default_coords_type = default_coords_type
        self._metric = metric

    @property
    def default_point_type(self):
        """Point type.

        `vector` or `matrix`.
        """
        if len(self.shape) == 1:
            return "vector"
        return "matrix"

    @abc.abstractmethod
    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., *shape]
            Point to evaluate.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the manifold.
        """

    @abc.abstractmethod
    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., *shape]
            Vector.
        base_point : array-like, shape=[..., *shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """

    @abc.abstractmethod
    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        Parameters
        ----------
        vector : array-like, shape=[..., *shape]
            Vector.
        base_point : array-like, shape=[..., *shape]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *shape]
            Tangent vector at base point.
        """

    @abc.abstractmethod
    def random_point(self, n_samples=1, bound=1.0):
        """Sample random points on the manifold.

        If the manifold is compact, a uniform distribution is used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample for non compact manifolds.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., *shape]
            Points sampled on the manifold.
        """

    def regularize(self, point):
        """Regularize a point to the canonical representation for the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., *shape]
            Point.

        Returns
        -------
        regularized_point : array-like, shape=[..., *shape]
            Regularized point.
        """
        regularized_point = point
        return regularized_point

    @property
    def metric(self):
        """Riemannian Metric associated to the Manifold."""
        return self._metric

    @metric.setter
    def metric(self, metric):
        if metric is not None:
            if not isinstance(metric, RiemannianMetric):
                raise ValueError("The argument must be a RiemannianMetric object")
            if metric.dim != self.dim:
                metric.dim = self.dim
        self._metric = metric

    def random_tangent_vec(self, base_point, n_samples=1):
        """Generate random tangent vec.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        base_point :  array-like, shape=[..., *shape]
            Point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *shape]
            Tangent vec at base point.
        """
        if (
            n_samples > 1
            and base_point.ndim > len(self.shape)
            and n_samples != len(base_point)
        ):
            raise ValueError(
                "The number of base points must be the same as the "
                "number of samples, when different from 1."
            )
        return gs.squeeze(
            self.to_tangent(
                gs.random.normal(size=(n_samples,) + self.shape), base_point
            )
        )

    def local_basis(self, base_point):
        """Local basis of the tangent space of the manifold at the point.

        It correspond to patial_i vector field for an implicit
        minimal system of coordinate x_i

        Parameters
        ----------
        base_point : array-like, shape=[..., *shape]
            Point.

        Returns
        -------
        local_basis : array-like, shape=[..., dim, *shape]
            Regularized point.
        """
        if not self.default_coords_type == "intrinsic":
            raise NotImplementedError(
                "Default local basis is only for intrinsic vector "
                "point case with shape (dim,)"
            )
        if base_point.shape == self.shape:
            return gs.eye(self.dim)
        return gs.broadcast_to(
            gs.expand_dims(gs.eye(self.dim), 0),
            (base_point.shape[0], self.dim, self.dim),
        )

    def local_gram(self, base_point, inverse=False, _local_basis=None):
        """Inverse gram matrix of the local basis.

        It is the gram in the sense of the canonical inner product of the shape
        representation choice as if it was embedded in Eucliden.
        It is just a matter of representation and does not impact any choices
        other than the local basis

        Parameters
        ----------
        base_point : array-like, shape=[..., *shape]
            Point.

        Returns
        -------
        local_gram : array-like, shape=[..., dim, dim]
            Regularized point.
        """
        if self.default_coords_type == "intrinsic":
            if base_point.shape == self.shape:
                return gs.eye(self.dim)
            return gs.broadcast_to(
                gs.expand_dims(gs.eye(self.dim), 0),
                (base_point.shape[0], self.dim, self.dim),
            )

        local_basis = (
            self.local_basis(base_point) if _local_basis is None else _local_basis
        )

        lb_flat = gs.reshape(local_basis, (-1, self.dim, self.shape_dim))
        gram = gs.einsum("...ik, ...jk -> ...ij", lb_flat, lb_flat)

        if inverse:
            gram = gs.linalg.inv(gram)

        if base_point.shape == self.shape:
            return gram[0]
        return gram

    def local_basis_representation(self, vector, base_point, _local_basis=None):
        """Express the vector in the local basis of the manifold.

        v=sum v^i e_i so <v,e_j> = sum v^i<e_i,e_j>
        [v^i] = G^{-1}Pv where P = [e_i] and G = [<e_i,e_j>] = PP^T

        Parameters
        ----------
        vector : array-like, shape=[..., *shape]
            Vector.
        base_point : array-like, shape=[..., *shape]
            Point on the manifold.

        Returns
        -------
        vector_intra : array-like, shape=[..., dim]
            Coordinate of the local vector in the locale basis
        """
        if self.default_coords_type == "intrinsic":
            return vector

        local_basis = (
            self.local_basis(base_point) if _local_basis is None else _local_basis
        )

        scal_repr = gs.einsum(
            "...k, ...jk -> ...j",
            gs.reshape(vector, (-1, self.shape_dim)),
            gs.reshape(local_basis, (-1, self.dim, self.shape_dim)),
        )
        inv_gram = self.local_gram(base_point, local_basis, inverse=True)

        lb_repr = gs.einsum(
            "...ij, ...j -> ...i",
            gs.reshape(inv_gram, (-1, self.dim, self.dim)),
            gs.reshape(scal_repr, (-1, self.dim)),
        )

        if base_point.shape == self.shape:
            return lb_repr[0]
        return lb_repr

    def local_shape_representation(self, vector_intra, base_point, _local_basis=None):
        """Given a vector express in local base, return.

        The vector in standard shape form
        v=sum v^i e_i = P[v^i] where P = [e_i]

        Parameters
        ----------
        vector_intra : array-like, shape=[..., dim]
            Vector in local basis representation.
        base_point : array-like, shape=[..., *shape]
            Point on the manifold.

        Returns
        -------
        vector : array-like, shape=[..., *shape]
            Coordinate of the local vector in the locale basis
        """
        if self.default_coords_type == "intrinsic":
            return vector_intra

        local_basis = (
            self.local_basis(base_point) if _local_basis is None else _local_basis
        )

        flat_repr = gs.einsum(
            "...i,il -> ...l",
            gs.reshape(vector_intra, (-1, self.dim)),
            gs.reshape(local_basis, (self.dim, self.shape_dim)),
        )
        shape_rept = gs.reshape(flat_repr, (-1,) + self.shape)

        if base_point.shape == self.shape:
            return shape_rept[0]
        return shape_rept

    def _local_basis_representation_vectorize(
        self, vector, base_point, _local_basis=None
    ):
        """Express the vector in the local basis of the manifold.

        WIP: Exemple of total factorization.
        v=sum v^i e_i so <v,e_j> = sum v^i<e_i,e_j>
        [v^i] = G^{-1}Pv where P = [e_i] and G = [<e_i,e_j>] = PP^T

        Parameters
        ----------
        vector : array-like, shape=[..., *shape]
            Vector.
        base_point : array-like, shape=[..., *shape]
            Point on the manifold.

        Returns
        -------
        vector_intra : array-like, shape=[..., dim]
            Coordinate of the local vector in the locale basis
        """
        if self.default_coords_type == "intrinsic":
            return vector

        rad = vector.shape[: -len(self.shape)]

        local_basis = (
            self.local_basis(base_point) if _local_basis is None else _local_basis
        )
        bc_local_basis = gs.broadcast_to(local_basis, rad + (self.dim,) + self.shape)

        scal_repr = gs.einsum(
            "...k, ...jk -> ...j",
            gs.reshape(vector, (-1, self.shape_dim)),
            gs.reshape(bc_local_basis, (-1, self.dim, self.shape_dim)),
        )
        inv_gram = self.local_gram(base_point, _local_basis, inverse=True)
        bc_inv_gram = gs.broadcast_to(inv_gram, rad + (self.dim, self.dim))

        lb_repr = gs.einsum(
            "...ij, ...j -> ...i",
            gs.reshape(bc_inv_gram, (-1, self.dim, self.dim)),
            gs.reshape(scal_repr, (-1, self.dim)),
        )

        return lb_repr.reshape(rad + (self.dim,))
