"""Abstract classes for manifolds.

Lead authors: Nicolas Guigui and Nina Miolane.
"""

import abc

import geomstats.backend as gs
from geomstats.geometry.complex_manifold import ComplexManifold
from geomstats.geometry.manifold import Manifold

CDTYPE = gs.get_default_cdtype()


class VectorSpace(Manifold, abc.ABC):
    """Abstract class for vector spaces.

    Parameters
    ----------
    shape : tuple
        Shape of the elements of the vector space. The dimension is the
        product of these values by default.
    """

    def __init__(self, shape, **kwargs):
        kwargs.setdefault("dim", int(gs.prod(gs.array(shape))))
        super().__init__(shape=shape, **kwargs)
        self.shape = shape
        self._basis = None

    def belongs(self, point, atol=gs.atol):
        """Evaluate if the point belongs to the vector space.

        This method checks the shape of the input point.

        Parameters
        ----------
        point : array-like, shape=[.., {dim, [n, n]}]
            Point to test.
        atol : float
            Unused here.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the space.
        """
        point = gs.array(point)
        minimal_ndim = len(self.shape)
        if self.shape[0] == 1 and len(point.shape) <= 1:
            point = gs.transpose(gs.to_ndarray(gs.to_ndarray(point, 1), 2))
        belongs = point.shape[-minimal_ndim:] == self.shape
        if point.ndim <= minimal_ndim:
            return belongs
        return gs.tile(gs.array([belongs]), [point.shape[0]])

    @staticmethod
    def projection(point):
        """Project a point to the vector space.

        This method is for compatibility and returns `point`. `point` should
        have the right shape,

        Parameters
        ----------
        point: array-like, shape[..., {dim, [n, n]}]
            Point.

        Returns
        -------
        point: array-like, shape[..., {dim, [n, n]}]
            Point.
        """
        return point

    def is_tangent(self, vector, base_point=None, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Tangent vectors are identified with points of the vector space so
        this checks the shape of the input vector.

        Parameters
        ----------
        vector : array-like, shape=[..., {dim, [n, n]}]
            Vector.
        base_point : array-like, shape=[..., {dim, [n, n]}]
            Point in the vector space.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        return self.belongs(vector, atol)

    def to_tangent(self, vector, base_point=None):
        """Project a vector to a tangent space of the vector space.

        This method is for compatibility and returns vector.

        Parameters
        ----------
        vector : array-like, shape=[..., {dim, [n, n]}]
            Vector.
        base_point : array-like, shape=[..., {dim, [n, n]}]
            Point in the vector space

        Returns
        -------
        tangent_vec : array-like, shape=[..., {dim, [n, n]}]
            Tangent vector at base point.
        """
        return self.projection(vector)

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

    @basis.setter
    def basis(self, basis):
        if len(basis) < self.dim:
            raise ValueError(
                "The basis should have length equal to the " "dimension of the space."
            )
        self._basis = basis

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

    def __init__(self, shape, **kwargs):
        kwargs.setdefault("dim", int(gs.prod(gs.array(shape))))
        super(ComplexVectorSpace, self).__init__(shape=shape, **kwargs)
        self.shape = shape
        self._basis = None

    def belongs(self, point, atol=gs.atol):
        """Evaluate if the point belongs to the vector space.

        This method checks the shape of the input point.

        Parameters
        ----------
        point : array-like, shape=[.., {dim, [n, n]}]
            Point to test.
        atol : float
            Unused here.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the space.
        """
        point = gs.array(point)
        minimal_ndim = len(self.shape)
        if self.shape[0] == 1 and len(point.shape) <= 1:
            point = gs.transpose(gs.to_ndarray(gs.to_ndarray(point, 1), 2))
        belongs = point.shape[-minimal_ndim:] == self.shape
        if point.ndim <= minimal_ndim:
            return belongs
        return gs.tile(gs.array([belongs]), [point.shape[0]])

    @staticmethod
    def projection(point):
        """Project a point to the vector space.

        This method is for compatibility and returns `point`. `point` should
        have the right shape,

        Parameters
        ----------
        point: array-like, shape[..., {dim, [n, n]}]
            Point.

        Returns
        -------
        point: array-like, shape[..., {dim, [n, n]}]
            Point.
        """
        return point

    def is_tangent(self, vector, base_point=None, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Tangent vectors are identified with points of the vector space so
        this checks the shape of the input vector.

        Parameters
        ----------
        vector : array-like, shape=[..., {dim, [n, n]}]
            Vector.
        base_point : array-like, shape=[..., {dim, [n, n]}]
            Point in the vector space.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        return self.belongs(vector, atol)

    def to_tangent(self, vector, base_point=None):
        """Project a vector to a tangent space of the vector space.

        This method is for compatibility and returns vector.

        Parameters
        ----------
        vector : array-like, shape=[..., {dim, [n, n]}]
            Vector.
        base_point : array-like, shape=[..., {dim, [n, n]}]
            Point in the vector space

        Returns
        -------
        tangent_vec : array-like, shape=[..., {dim, [n, n]}]
            Tangent vector at base point.
        """
        return self.projection(vector)

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
        point : array-like, shape=[..., dim]
           Sample.
        """
        size = self.shape
        if n_samples != 1:
            size = (n_samples,) + self.shape
        point = gs.cast(gs.random.rand(*size), dtype=CDTYPE) - 0.5
        point += 1j * (gs.cast(gs.random.rand(*size), dtype=CDTYPE) - 0.5)
        point *= 2 * bound
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
                "The basis should have length equal to the " "dimension of the space."
            )
        self._basis = basis

    @abc.abstractmethod
    def _create_basis(self):
        """Create a canonical basis."""


class LevelSet(Manifold, abc.ABC):
    """Class for manifolds embedded in a vector space by a submersion.

    Parameters
    ----------
    dim : int
        Dimension of the embedded manifold.
    embedding_space : VectorSpace
        Embedding space.
    default_coords_type : str, {'intrinsic', 'extrinsic', etc}
        Coordinate type.
        Optional, default: 'intrinsic'.
    """

    def __init__(
        self,
        dim,
        embedding_space,
        submersion,
        value,
        tangent_submersion,
        default_coords_type="intrinsic",
        **kwargs
    ):
        kwargs.setdefault("shape", embedding_space.shape)
        super().__init__(dim=dim, default_coords_type=default_coords_type, **kwargs)
        self.embedding_space = embedding_space
        self.embedding_metric = embedding_space.metric
        self.submersion = submersion
        if isinstance(value, float):
            value = gs.array(value)
        self.value = value
        self.tangent_submersion = tangent_submersion

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
        belongs = self.embedding_space.belongs(point, atol)
        if not gs.any(belongs):
            return belongs
        value = self.value
        constraint = gs.isclose(self.submersion(point), value, atol=atol)
        if value.ndim == 2:
            constraint = gs.all(constraint, axis=(-2, -1))
        elif value.ndim == 1:
            constraint = gs.all(constraint, axis=-1)
        return gs.logical_and(belongs, constraint)

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
        belongs = self.embedding_space.is_tangent(vector, base_point, atol)
        tangent_sub_applied = self.tangent_submersion(vector, base_point)
        constraint = gs.isclose(tangent_sub_applied, 0.0, atol=atol)
        value = self.value
        if value.ndim == 2:
            constraint = gs.all(constraint, axis=(-2, -1))
        elif value.ndim == 1:
            constraint = gs.all(constraint, axis=-1)
        return gs.logical_and(belongs, constraint)

    def intrinsic_to_extrinsic_coords(self, point_intrinsic):
        """Convert from intrinsic to extrinsic coordinates.

        Parameters
        ----------
        point_intrinsic : array-like, shape=[..., dim]
            Point in the embedded manifold in intrinsic coordinates.

        Returns
        -------
        point_extrinsic : array-like, shape=[..., dim_embedding]
            Point in the embedded manifold in extrinsic coordinates.
        """
        raise NotImplementedError("intrinsic_to_extrinsic_coords is not implemented.")

    def extrinsic_to_intrinsic_coords(self, point_extrinsic):
        """Convert from extrinsic to intrinsic coordinates.

        Parameters
        ----------
        point_extrinsic : array-like, shape=[..., dim_embedding]
            Point in the embedded manifold in extrinsic coordinates,
            i. e. in the coordinates of the embedding manifold.

        Returns
        -------
        point_intrinsic : array-lie, shape=[..., dim]
            Point in the embedded manifold in intrinsic coordinates.
        """
        raise NotImplementedError("extrinsic_to_intrinsic_coords is not implemented.")

    @abc.abstractmethod
    def projection(self, point):
        """Project a point in embedding manifold on embedded manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim_embedding]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., dim_embedding]
            Projected point.
        """

    @abc.abstractmethod
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
        """


class OpenSet(Manifold, abc.ABC):
    """Class for manifolds that are open sets of a vector space.

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

    def __init__(self, dim, embedding_space, **kwargs):
        kwargs.setdefault("shape", embedding_space.shape)
        super().__init__(dim=dim, **kwargs)
        self.embedding_space = embedding_space

    def is_tangent(self, vector, base_point=None, atol=gs.atol):
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
        return self.embedding_space.belongs(vector, atol)

    def to_tangent(self, vector, base_point=None):
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
        """
        return self.embedding_space.projection(vector)

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
        samples : array-like, shape=[..., {dim, [n, n]}]
            Points sampled on the hypersphere.
        """
        sample = self.embedding_space.random_point(n_samples, bound)
        return self.projection(sample)

    @abc.abstractmethod
    def projection(self, point):
        """Project a point in embedding manifold on manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., dim]
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

    def __init__(self, dim, embedding_space, **kwargs):
        kwargs.setdefault("shape", embedding_space.shape)
        super().__init__(dim=dim, **kwargs)
        self.embedding_space = embedding_space

    def is_tangent(self, vector, base_point=None, atol=gs.atol):
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
        return self.embedding_space.belongs(vector, atol)

    def to_tangent(self, vector, base_point=None):
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
        """
        return self.embedding_space.projection(vector)

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
        samples : array-like, shape=[..., {dim, [n, n]}]
            Points sampled on the hypersphere.
        """
        sample = self.embedding_space.random_point(n_samples, bound)
        return self.projection(sample)

    @abc.abstractmethod
    def projection(self, point):
        """Project a point in embedding manifold on manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., dim]
            Projected point.
        """
