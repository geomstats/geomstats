"""Abstract classes for manifolds.

Lead authors: Nicolas Guigui and Nina Miolane.
"""

import abc
import math

import geomstats.backend as gs
from geomstats.geometry.complex_manifold import ComplexManifold
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.pullback_metric import PullbackMetric
from geomstats.vectorization import get_batch_shape


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
        point : array-like, shape=[..., dim]
           Sample.
        """
        size = (self.dim,)
        if n_samples != 1:
            size = (n_samples,) + size
        return bound * (gs.random.rand(*size) - 0.5) * 2

    def random_tangent_vec(self, base_point=None, n_samples=1):
        """Generate random tangent vec.

        This method is not recommended for statistical purposes, as the
        tangent vectors generated are not drawn from a distribution related
        to the Riemannian metric.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        base_point :  array-like, shape={[n_samples, *point_shape], [*point_shape,]}
            Point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *point_shape]
            Tangent vec at base point.
        """
        if base_point is None:
            return self.random_point(n_samples)

        if (
            n_samples > 1
            and base_point.ndim > self.point_ndim
            and n_samples != base_point.shape[0]
        ):
            raise ValueError(
                "The number of base points must be the same as the "
                "number of samples, when the number of base points is different from 1."
            )
        if n_samples == 1 and base_point.ndim > self.point_ndim:
            n_samples = base_point.shape[0]
        return self.random_point(n_samples)

    @property
    def basis(self):
        """Basis of the vector space."""
        if self._basis is None:
            self._basis = self._create_basis()
        return self._basis

    @abc.abstractmethod
    def _create_basis(self):
        """Create a canonical basis."""


class MatrixVectorSpace(VectorSpace, abc.ABC):
    """A matrix vector space."""

    @abc.abstractmethod
    def basis_representation(self, matrix_representation):
        """Compute the coefficients of matrices in the given basis.

        This takes a matrix (the matrix representation of a point) and
        transforms it into its corresponding vector representation
        (the coefficients wrt a given basis).

        Previously, this method was called `to_vector`. `basis_representation`
        makes it more clear that the vector representation depends on the chosen
        basis.

        Parameters
        ----------
        matrix_representation : array-like, shape=[..., *point_shape]
            Matrix.

        Returns
        -------
        basis_representation : array-like, shape=[..., dim]
            Coefficients in the basis.
        """
        raise NotImplementedError("basis_representation not implemented.")

    def matrix_representation(self, basis_representation):
        """Compute the matrix representation for the given basis coefficients.

        This takes a vector representation of a point (the coefficients wrt
        a given basis) and creates the corresponding matrix representation.

        Parameters
        ----------
        basis_representation : array-like, shape=[..., dim]
            Coefficients in the basis.

        Returns
        -------
        matrix_representation : array-like, shape=[..., *point_shape]
            Matrix.
        """
        return gs.einsum("...i,ijk ->...jk", basis_representation, self.basis)

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
        return self.matrix_representation(super().random_point(n_samples, bound))


class ComplexVectorSpace(ComplexManifold, abc.ABC):
    """Abstract class for complex vector spaces.

    Parameters
    ----------
    shape : tuple
        Shape of the elements of the vector space. The dimension is the
        product of these values by default.
    """

    def __init__(self, shape, dim=None, **kwargs):
        if dim is None:
            dim = math.prod(shape)
        super().__init__(shape=shape, dim=dim, **kwargs)

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


class ComplexMatrixVectorSpace(ComplexVectorSpace):
    """A matrix vector space."""


class LevelSet(Manifold, abc.ABC):
    """Class for manifolds embedded in a vector space by a submersion.

    Parameters
    ----------
    intrinsic : bool
        Coordinates type.
    """

    def __init__(self, intrinsic=False, shape=None, **kwargs):
        self.embedding_space = self._define_embedding_space()

        if shape is None:
            shape = self.embedding_space.shape

        super().__init__(intrinsic=intrinsic, shape=shape, **kwargs)

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

        n_batch = len(get_batch_shape(self.point_ndim, base_point, vector))
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


class OpenSet(Manifold, abc.ABC):
    """Class for manifolds that are open sets.

    NB: if the embedding space is a vector space, use `VectorSpaceOpenSet`.

    Parameters
    ----------
    embedding_space: Manifold
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
        return self.embedding_space.is_tangent(vector, base_point, atol)

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
        return self.embedding_space.to_tangent(vector, base_point)

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


class VectorSpaceOpenSet(OpenSet, abc.ABC):
    """Class for manifolds that are open sets of a vector space.

    In this case, tangent vectors are identified with vectors of the embedding
    space.

    Parameters
    ----------
    embedding_space: VectorSpace
        Embedding space that contains the manifold.
    """

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


class ComplexVectorSpaceOpenSet(ComplexManifold, abc.ABC):
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
        super().__init__(shape=shape, intrinsic=False, **kwargs)
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
        super().__init__(dim=dim, shape=(dim,), intrinsic=True, equip=equip)
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
        return gs.einsum("...ij,...j->...i", jacobian_immersion, tangent_vec)

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


class DiffeomorphicManifold(Manifold):
    """A manifold defined by a diffeomorphism."""

    def __init__(self, diffeo, image_space, **kwargs):
        self.diffeo = diffeo
        self.image_space = image_space
        super().__init__(**kwargs)

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
        if not self.intrinsic:
            raise ValueError("`belongs` is not implemented.")
        return self.image_space.belongs(self.diffeo(point), atol=atol)

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
        if not self.intrinsic:
            raise ValueError("`is_tangent` is not implemented.")

        image_point = self.diffeo(base_point)
        image_vector = self.diffeo.tangent(
            vector, base_point=base_point, image_point=image_point
        )
        return self.image_space.is_tangent(image_vector, image_point, atol=atol)

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
        image_point = self.diffeo(base_point)
        image_vector = self.diffeo.tangent(
            vector, base_point=base_point, image_point=image_point
        )
        image_tangent_vec = self.image_space.to_tangent(image_vector, image_point)
        return self.diffeo.inverse_tangent(
            image_tangent_vec, image_point=image_point, base_point=base_point
        )

    def random_point(self, n_samples=1, **kwargs):
        """Sample random points on the manifold according to some distribution.

        If the manifold is compact, preferably a uniform distribution will be used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., *point_shape]
            Points sampled on the manifold.
        """
        image_point = self.image_space.random_point(n_samples=n_samples, **kwargs)
        return self.diffeo.inverse(image_point)

    def regularize(self, point):
        """Regularize a point to the canonical representation for the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point.

        Returns
        -------
        regularized_point : array-like, shape=[..., *point_shape]
            Regularized point.
        """
        image_point = self.diffeo(point)
        regularized_image_point = self.image_space.regularize(image_point)
        return self.diffeo.inverse(regularized_image_point)

    def random_tangent_vec(self, base_point=None, n_samples=1):
        """Generate random tangent vec.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        base_point :  array-like, shape={[n_samples, *point_shape], [*point_shape,]}
            Point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., *point_shape]
            Tangent vec at base point.
        """
        image_point = self.diffeo(base_point)
        image_tangent_vec = self.image_space.random_tangent_vec(
            image_point, n_samples=n_samples
        )
        return self.diffeo.inverse_tangent(
            image_tangent_vec, image_point=image_point, base_point=base_point
        )


class DiffeomorphicVectorSpace(VectorSpace, DiffeomorphicManifold):
    """A vector space defined by a diffeomorphism."""

    def projection(self, point):
        r"""Make a matrix null-row-sum symmetric.

        It considers only the first :math:`n-1 \times n-1` components.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        sym : array-like, shape=[..., n, n]
            Symmetric matrix.
        """
        image_point = self.diffeo(point)
        proj_image_point = self.image_space.projection(image_point)
        return self.diffeo.inverse(proj_image_point)


class DiffeomorphicMatrixVectorSpace(MatrixVectorSpace, DiffeomorphicVectorSpace):
    """A matrix vector space defined by a diffeomorphism."""

    def basis_representation(self, matrix_representation):
        """Convert a symmetric matrix into a vector.

        Parameters
        ----------
        matrix_representation : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        basis_representation : array-like, shape=[..., n(n+1)/2]
            Vector.
        """
        image_matrix_representation = self.diffeo(matrix_representation)
        return self.image_space.basis_representation(image_matrix_representation)

    def matrix_representation(self, basis_representation):
        """Convert a vector into a symmetric matrix.

        Parameters
        ----------
        basis_representation : array-like, shape=[..., n(n+1)/2]
            Vector.

        Returns
        -------
        matrix_representation : array-like, shape=[..., n, n]
            Symmetric matrix.
        """
        image_point = self.image_space.matrix_representation(basis_representation)
        return self.diffeo.inverse(image_point)
