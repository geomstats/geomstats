"""Manifold module.

In other words, a topological space that locally resembles
Euclidean space near each point.

Lead author: Nina Miolane.
"""

import abc

import geomstats.backend as gs
import geomstats.errors


class Manifold(abc.ABC):
    r"""Class for manifolds.

    Parameters
    ----------
    dim : int
        Dimension of the manifold.
    shape : tuple of int
        Shape of one element of the manifold.
        Optional, default : None.
    default_coords_type : str, {'intrinsic', 'extrinsic', etc}
        Coordinate type.
        Optional, default: 'intrinsic'.
    equip : bool
        If True, equip space with default metric.

    Attributes
    ----------
    point_ndim : int
        Dimension of point array.
    default_point_type : str
        Point type: "vector" or "matrix".
    """

    def __init__(
        self,
        dim,
        shape,
        default_coords_type="intrinsic",
        equip=True,
    ):
        geomstats.errors.check_integer(dim, "dim")

        if not isinstance(shape, tuple):
            raise ValueError("Expected a tuple for the shape argument.")

        self.dim = dim
        self.shape = shape
        self.default_coords_type = default_coords_type

        self.point_ndim = len(self.shape)
        if self.point_ndim == 1:
            self.default_point_type = "vector"
        elif self.point_ndim == 2:
            self.default_point_type = "matrix"
        else:
            self.default_point_type = "other"

        if equip:
            self.equip_with_metric()

    def equip_with_metric(self, Metric=None, **metric_kwargs):
        """Equip manifold with a Riemannian metric.

        Parameters
        ----------
        Metric : RiemannianMetric object
            If None, default metric will be used.
        """
        if Metric is None:
            out = self.default_metric()
            if isinstance(out, tuple):
                Metric, kwargs = out
                kwargs.update(metric_kwargs)
                metric_kwargs = kwargs
            else:
                Metric = out

        self.metric = Metric(self, **metric_kwargs)

    def equip_with_group_action(self, group_action):
        """Equip manifold with group action.

        Parameters
        ----------
        group_action : str
            Group action.
        """
        self.group_action = group_action

    def _check_equip_with_quotient_structure(self):
        if not hasattr(self, "_quotient_map"):
            raise ValueError("No quotient structure defined for this manifold.")

        for structure_name in ("metric", "group_action"):
            if not hasattr(self, structure_name):
                raise ValueError(f"Need to equip with `{structure_name}` first")

    def equip_with_quotient_structure(self):
        """Equip manifold with quotient structure.

        Creates attributes `quotient` and `fiber_bundle`.
        """
        self._check_equip_with_quotient_structure()

        key = type(self.metric), self.group_action

        out = self._quotient_map.get(key, None)
        if out is None:
            raise ValueError(f"No mapping for key: {key}")
        FiberBundle_, QuotientMetric_ = out

        self.fiber_bundle = FiberBundle_(total_space=self)

        self.quotient = self.new(equip=False)
        self.quotient.equip_with_metric(QuotientMetric_, fiber_bundle=self.fiber_bundle)

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    @abc.abstractmethod
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
        return gs.copy(point)

    def random_tangent_vec(self, base_point, n_samples=1):
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
        if (
            n_samples > 1
            and base_point.ndim > len(self.shape)
            and n_samples != len(base_point)
        ):
            raise ValueError(
                "The number of base points must be the same as the "
                "number of samples, when the number of base points is different from 1."
            )
        batch_size = () if n_samples == 1 else (n_samples,)
        return self.to_tangent(
            gs.random.normal(size=batch_size + self.shape), base_point
        )
