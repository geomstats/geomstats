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
        if belongs:
            return gs.ones(point.shape[:-minimal_ndim], dtype=bool)
        return gs.zeros(point.shape[:-minimal_ndim], dtype=bool)

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
        if belongs:
            return gs.ones(point.shape[:-minimal_ndim], dtype=bool)
        return False

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
