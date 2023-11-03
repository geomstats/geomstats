"""Abstract classes for manifolds.

Lead authors: Nicolas Guigui and Nina Miolane.
"""

import abc
import math

import geomstats.backend as gs
from geomstats.geometry.complex_manifold import ComplexManifold
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.pullback_metric import PullbackMetric


class VectorSpace(Manifold, abc.ABC):
    """Abstract class for vector spaces.

    Parameters
    ----------
    shape : tuple
        Shape of the elements of the vector space. The dimension is the
        product of these values by default.
    """

    def __init__(self, shape, dim=None, **kwargs):
        if dim is None:
            dim = math.prod(shape)
        super().__init__(dim=dim, shape=shape, **kwargs)
        self._basis = None

    def belongs(self, point, atol=gs.atol):
        """Evaluate if the point belongs to the vector space.

        This method checks the shape of the input point.

        Parameters
        ----------
        point : array-like, shape=[.., *point_shape]
            Point to test.
        atol : float
            Unused here.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the space.
        """
        belongs = self.shape == point.shape[-self.point_ndim :]
        shape = point.shape[: -self.point_ndim]
        if belongs:
            return gs.ones(shape, dtype=bool)
        return gs.zeros(shape, dtype=bool)

    @staticmethod
    def projection(point):
        """Project a point to the vector space.

        This method is for compatibility and returns `point`. `point` should
        have the right shape,

        Parameters
        ----------
        point: array-like, shape[..., *point_shape]
            Point.

        Returns
        -------
        point: array-like, shape[..., *point_shape]
            Point.
        """
        return gs.copy(point)

    def is_tangent(self, vector, base_point=None, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Tangent vectors are identified with points of the vector space so
        this checks the shape of the input vector.

        Parameters
        ----------
        vector : array-like, shape=[..., *point_shape]
            Vector.
        base_point : array-like, shape=[..., *point_shape]
            Point in the vector space.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : array-like, shape=[...,]
            Boolean denoting if vector is a tangent vector at the base point.
        """
        belongs = self.belongs(vector, atol)
        if base_point is not None and base_point.ndim > vector.ndim:
            return gs.broadcast_to(belongs, base_point.shape[: -self.point_ndim])
        return belongs

    def to_tangent(self, vector, base_point=None):
        """Project a vector to a tangent space of the vector space.

        This method is for compatibility and returns vector.

        Parameters
        ----------
        vector : array-like, shape=[..., *point_shape]
            Vector.
        base_point : array-like, shape=[..., *point_shape]
            Point in the vector space

        Returns
        -------
        tangent_vec : array-like, shape=[..., *point_shape]
            Tangent vector at base point.
        """
        tangent_vec = self.projection(vector)
        if base_point is not None and base_point.ndim > vector.ndim:
            return gs.broadcast_to(tangent_vec, base_point.shape)
        return tangent_vec

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in the vector space with a uniform distribution in a box.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Side of hypercube support of the uniform distribution.
            Optional, default: 1.0

        Returns
        -------
        point : array-like, shape=[..., *point_shape]
           Sample.
        """
        size = self.shape
        if n_samples != 1:
            size = (n_samples,) + self.shape
        point = bound * (gs.random.rand(*size) - 0.5) * 2
        return point

    @property
    def basis(self):
        """Basis of the vector space."""
        if self._basis is None:
            self._basis = self._create_basis()
        return self._basis

    @abc.abstractmethod
    def _create_basis(self):
        """Create a canonical basis."""


class ComplexVectorSpace(ComplexManifold, abc.ABC):
    """Abstract class for complex vector spaces.

    Parameters
    ----------
    shape : tuple
        Shape of the elements of the vector space. The dimension is the
        product of these values by default.
    default_point_type : str, {'vector', 'matrix'}
        Point type.
        Optional, default: 'vector'.
    """

    def __init__(self, shape, dim=None, **kwargs):
        if dim is None:
            dim = math.prod(shape)
        super().__init__(shape=shape, dim=dim, **kwargs)
        self._basis = None

    def belongs(self, point, atol=gs.atol):
        """Evaluate if the point belongs to the vector space.

        This method checks the shape of the input point.

        Parameters
        ----------
        point : array-like, shape=[.., *point_shape]
            Point to test.
        atol : float
            Unused here.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the space.
        """
        belongs = self.shape == point.shape[-self.point_ndim :]
        shape = point.shape[: -self.point_ndim]
        if belongs:
            return gs.ones(shape, dtype=bool)
        return gs.zeros(shape, dtype=bool)

    @staticmethod
    def projection(point):
        """Project a point to the vector space.

        This method is for compatibility and returns `point`. `point` should
        have the right shape,

        Parameters
        ----------
        point: array-like, shape[..., *point_shape]
            Point.

        Returns
        -------
        point: array-like, shape[..., *point_shape]
            Point.
        """
        return gs.copy(point)

    def is_tangent(self, vector, base_point=None, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Tangent vectors are identified with points of the vector space so
        this checks the shape of the input vector.

        Parameters
        ----------
        vector : array-like, shape=[..., *point_shape]
            Vector.
        base_point : array-like, shape=[..., *point_shape]
            Point in the vector space.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        belongs = self.belongs(vector, atol)
        if base_point is not None and base_point.ndim > vector.ndim:
            return gs.broadcast_to(belongs, base_point.shape[: -self.point_ndim])
        return belongs

    def to_tangent(self, vector, base_point=None):
        """Project a vector to a tangent space of the vector space.

        This method is for compatibility and returns vector.

        Parameters
        ----------
        vector : array-like, shape=[..., *point_shape]
            Vector.
        base_point : array-like, shape=[..., *point_shape]
            Point in the vector space

        Returns
        -------
        tangent_vec : array-like, shape=[..., *point_shape]
            Tangent vector at base point.
        """
        tangent_vec = self.projection(vector)
        if base_point is not None and base_point.ndim > vector.ndim:
            return gs.broadcast_to(tangent_vec, base_point.shape)
        return tangent_vec

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in the complex vector space with a uniform distribution in a box.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Side of hypercube support of the uniform distribution.
            Optional, default: 1.0

        Returns
        -------
        point : array-like, shape=[..., *point_shape]
           Sample.
        """
        size = self.shape
        if n_samples != 1:
            size = (n_samples,) + self.shape
        point = bound * (
            gs.random.rand(*size, dtype=gs.get_default_cdtype()) - 0.5 - 0.5j
        )
        return point

    @property
    def basis(self):
        """Basis of the vector space."""
        if self._basis is None:
            self._basis = self._create_basis()
        return self._basis

    @basis.setter
    def basis(self, basis):
        if len(basis) < self.dim:
            raise ValueError(
                "The basis should have length equal to the dimension of the space."
            )
        self._basis = basis

    @abc.abstractmethod
    def _create_basis(self):
        """Create a canonical basis."""


class LevelSet(Manifold, abc.ABC):
    """Class for manifolds embedded in a vector space by a submersion.

    Parameters
    ----------
    default_coords_type : str, {'intrinsic', 'extrinsic', etc}
        Coordinate type.
        Optional, default: 'extrinsic'.
    """

    def __init__(self, default_coords_type="extrinsic", shape=None, **kwargs):
        self.embedding_space = self._define_embedding_space()

        if shape is None:
            shape = self.embedding_space.shape

        super().__init__(default_coords_type=default_coords_type, shape=shape, **kwargs)

    @abc.abstractmethod
    def _define_embedding_space(self):
        """Define embedding space of the manifold.

        Returns
        -------
        embedding_space : Manifold
            Instance of Manifold.
        """

    @abc.abstractmethod
    def submersion(self, point):
        r"""Submersion that defines the manifold.

        :math:`\mathrm{submersion}(x)=0` defines the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., *point_shape]

        Returns
        -------
        submersed_point : array-like
        """

    @abc.abstractmethod
    def tangent_submersion(self, vector, point):
        """Tangent submersion.

        Parameters
        ----------
        vector : array-like, shape=[..., *point_shape]
        point : array-like, shape=[..., *point_shape]

        Returns
        -------
        submersed_vector : array-like
        """

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., *point_shape]
            Point to evaluate.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the manifold.
        """
        belongs = self.embedding_space.belongs(point, atol)
        if not gs.any(belongs):
            return belongs

        submersed_point = self.submersion(point)

        n_batch = gs.ndim(point) - len(self.shape)
        axis = tuple(range(-len(submersed_point.shape) + n_batch, 0))

        if gs.is_complex(submersed_point):
            constraint = gs.isclose(submersed_point, 0.0 + 0.0j, atol=atol)
        else:
            constraint = gs.isclose(submersed_point, 0.0, atol=atol)

        if axis:
            constraint = gs.all(constraint, axis=axis)

        return gs.logical_and(belongs, constraint)

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., *point_shape]
            Vector.
        base_point : array-like, shape=[..., *point_shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        belongs = self.embedding_space.is_tangent(vector, base_point, atol)
        if not gs.any(belongs):
            return belongs

        submersed_vector = self.tangent_submersion(vector, base_point)

        n_batch = max(gs.ndim(base_point), gs.ndim(vector)) - len(self.shape)
        axis = tuple(range(-len(submersed_vector.shape) + n_batch, 0))

        constraint = gs.isclose(submersed_vector, 0.0, atol=atol)
        if axis:
            constraint = gs.all(constraint, axis=axis)

        return gs.logical_and(belongs, constraint)

    def intrinsic_to_extrinsic_coords(self, point_intrinsic):
        """Convert from intrinsic to extrinsic coordinates.

        Parameters
        ----------
        point_intrinsic : array-like, shape=[..., *point_shape]
            Point in the embedded manifold in intrinsic coordinates.

        Returns
        -------
        point_extrinsic : array-like, shape=[..., *embedding_space.point_shape]
            Point in the embedded manifold in extrinsic coordinates.
        """
        raise NotImplementedError("intrinsic_to_extrinsic_coords is not implemented.")

    def extrinsic_to_intrinsic_coords(self, point_extrinsic):
        """Convert from extrinsic to intrinsic coordinates.

        Parameters
        ----------
        point_extrinsic : array-like, shape=[..., *embedding_space.point_shape]
            Point in the embedded manifold in extrinsic coordinates,
            i. e. in the coordinates of the embedding manifold.

        Returns
        -------
        point_intrinsic : array-lie, shape=[..., *point_shape]
            Point in the embedded manifold in intrinsic coordinates.
        """
        raise NotImplementedError("extrinsic_to_intrinsic_coords is not implemented.")

    @abc.abstractmethod
    def projection(self, point):
        """Project a point in embedding manifold on embedded manifold.

        Parameters
        ----------
        point : array-like, shape=[..., *embedding_space.point_shape]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., *point_shape]
            Projected point.
        """

    @abc.abstractmethod
    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        Parameters
        ----------
        vector : array-like, shape=[..., *point_shape]
            Vector.
        base_point : array-like, shape=[..., *point_shape]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *point_shape]
            Tangent vector at base point.
        """


class OpenSet(Manifold, abc.ABC):
    """Class for manifolds that are open sets of a vector space.

    In this case, tangent vectors are identified with vectors of the embedding
    space.

    Parameters
    ----------
    embedding_space: VectorSpace
        Embedding space that contains the manifold.
    """

    def __init__(self, embedding_space, shape=None, **kwargs):
        self.embedding_space = embedding_space
        if shape is None:
            shape = embedding_space.shape
        super().__init__(shape=shape, **kwargs)

    def is_tangent(self, vector, base_point=None, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., *point_shape]
            Vector.
        base_point : array-like, shape=[..., *point_shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        is_tangent = self.embedding_space.belongs(vector, atol)
        if base_point is not None and base_point.ndim > vector.ndim:
            return gs.broadcast_to(is_tangent, base_point.shape[: -self.point_ndim])
        return is_tangent

    def to_tangent(self, vector, base_point=None):
        """Project a vector to a tangent space of the manifold.

        Parameters
        ----------
        vector : array-like, shape=[..., *point_shape]
            Vector.
        base_point : array-like, shape=[..., *point_shape]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *point_shape]
            Tangent vector at base point.
        """
        tangent_vec = self.embedding_space.projection(vector)
        if base_point is not None and base_point.ndim > vector.ndim:
            return gs.broadcast_to(tangent_vec, base_point.shape)
        return tangent_vec

    def random_point(self, n_samples=1, bound=1.0):
        """Sample random points on the manifold.

        Points are sampled from the embedding space using the distribution set
        for that manifold and then projected to the manifold. As a result, this
        is not a uniform distribution on the manifold itself.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample for the embedding space.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., *point_shape]
            Points sampled on the hypersphere.
        """
        sample = self.embedding_space.random_point(n_samples, bound)
        return self.projection(sample)

    @abc.abstractmethod
    def projection(self, point):
        """Project a point in embedding manifold on manifold.

        Parameters
        ----------
        point : array-like, shape=[..., *point_shape]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., *point_shape]
            Projected point.
        """


class ComplexOpenSet(ComplexManifold, abc.ABC):
    """Class for manifolds that are open sets of a complex vector space.

    In this case, tangent vectors are identified with vectors of the embedding
    space.

    Parameters
    ----------
    dim: int
        Dimension of the manifold. It is often the same as the embedding space
        dimension but may differ in some cases.
    embedding_space: VectorSpace
        Embedding space that contains the manifold.
    """

    def __init__(self, embedding_space, shape=None, **kwargs):
        if shape is None:
            shape = embedding_space.shape
        super().__init__(shape=shape, default_coords_type="extrinsic", **kwargs)
        self.embedding_space = embedding_space

    def is_tangent(self, vector, base_point=None, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., *point_shape]
            Vector.
        base_point : array-like, shape=[..., *point_shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        is_tangent = self.embedding_space.belongs(vector, atol)
        if base_point is not None and base_point.ndim > vector.ndim:
            return gs.broadcast_to(is_tangent, base_point.shape[: -self.point_ndim])
        return is_tangent

    def to_tangent(self, vector, base_point=None):
        """Project a vector to a tangent space of the manifold.

        Parameters
        ----------
        vector : array-like, shape=[..., *point_shape]
            Vector.
        base_point : array-like, shape=[..., *point_shape]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *point_shape]
            Tangent vector at base point.
        """
        tangent_vec = self.embedding_space.projection(vector)
        if base_point is not None and base_point.ndim > vector.ndim:
            return gs.broadcast_to(tangent_vec, base_point.shape)
        return tangent_vec

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
        samples : array-like, shape=[..., *point_shape]
            Points sampled on the hypersphere.
        """
        sample = self.embedding_space.random_point(n_samples, bound)
        return self.projection(sample)

    @abc.abstractmethod
    def projection(self, point):
        """Project a point in embedding manifold on manifold.

        Parameters
        ----------
        point : array-like, shape=[..., *point_shape]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., *point_shape]
            Projected point.
        """


class ImmersedSet(Manifold, abc.ABC):
    """Class for manifolds embedded in a vector space by an immersion.

    The manifold is represented with intrinsic coordinates, such that
    the immersion gives a parameterization of the manifold in these
    coordinates.

    Parameters
    ----------
    dim : int
        Dimension of the embedded manifold.
    """

    def __init__(self, dim, equip=True):
        super().__init__(
            dim=dim, shape=(dim,), default_coords_type="intrinsic", equip=equip
        )
        self.embedding_space = self._define_embedding_space()

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return PullbackMetric

    @abc.abstractmethod
    def _define_embedding_space(self):
        """Define embedding space of the manifold.

        Returns
        -------
        embedding_space : Manifold
            Instance of Manifold.
        """

    @abc.abstractmethod
    def immersion(self, point):
        """Evaluate the immersion function at a point.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in the immersed manifold.

        Returns
        -------
        immersion : array-like, shape=[..., dim_embedding]
            Immersion of the point.
        """

    def tangent_immersion(self, tangent_vec, base_point):
        """Evaluate the tangent immersion at a tangent vec and point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
        base_point : array-like, shape=[..., dim]
            Point in the immersed manifold.

        Returns
        -------
        tangent_vec_emb : array-like, shape=[..., dim_embedding]
        """
        jacobian_immersion = self.jacobian_immersion(base_point)
        return gs.matvec(jacobian_immersion, tangent_vec)

    def jacobian_immersion(self, base_point):
        """Evaluate the Jacobian of the immersion at a point.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point in the immersed manifold.

        Returns
        -------
        jacobian_immersion : array-like, shape=[..., dim_embedding, dim]
        """
        return gs.autodiff.jacobian_vec(self.immersion)(base_point)

    def hessian_immersion(self, base_point):
        """Compute the Hessian of the immersion.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        hessian_immersion : array-like, shape=[..., embedding_dim, dim, dim]
            Hessian at the base point
        """
        return gs.autodiff.hessian_vec(
            self.immersion, func_out_ndim=self.embedding_space.dim
        )(base_point)

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

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
        raise NotImplementedError("`is_tangent` is not implemented yet")

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to evaluate.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the manifold.
        """
        raise NotImplementedError("`is_tangent` is not implemented yet")

    def projection(self, point):
        """Project a point to the embedded manifold.

        This is simply point, since we are in intrinsic coordinates.

        Parameters
        ----------
        point : array-like, shape=[..., dim_embedding]
            Point in the embedding manifold.

        Returns
        -------
        projected_point : array-like, shape=[..., dim]
            Point in the embedded manifold.
        """
        raise NotImplementedError("`projection` is not implemented yet")

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        This is simply the vector since we are in intrinsic coordinates.

        Parameters
        ----------
        vector : array-like, shape=[..., dim_embedding]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point in the embedded manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        raise NotImplementedError("`to_tangent` is not implemented yet")

    def random_point(self, n_samples=1, bound=1.0):
        """Sample random points on the manifold according to some distribution.

        If the manifold is compact, preferably a uniform distribution will be used.

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
        samples : array-like, shape=[..., *point_shape]
            Points sampled on the manifold.
        """
        raise NotImplementedError("`random_point` is not implemented yet")
