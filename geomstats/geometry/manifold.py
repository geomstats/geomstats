"""Manifold module.

In other words, a topological space that locally resembles
Euclidean space near each point.

Lead author: Nina Miolane.
"""

import abc
import inspect
import types

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.quotient_metric import QuotientMetric


class Manifold(abc.ABC):
    r"""Class for manifolds.

    Parameters
    ----------
    dim : int
        Dimension of the manifold.
    shape : tuple of int
        Shape of one element of the manifold.
        Optional, default : None.
    intrinsic : bool
        Coordinate type.
    equip : bool
        If True, equip space with default metric.

    Attributes
    ----------
    point_ndim : int
        Dimension of point array.
    """

    def __init__(
        self,
        dim,
        shape,
        intrinsic=True,
        equip=True,
    ):
        geomstats.errors.check_integer(dim, "dim")

        if not isinstance(shape, tuple):
            raise ValueError("Expected a tuple for the shape argument.")

        self.dim = dim
        self.shape = shape
        self.intrinsic = intrinsic

        self.point_ndim = len(self.shape)

        if equip:
            self.equip_with_metric()

    def equip_with_metric(self, Metric=None, **metric_kwargs):
        """Equip manifold with a Riemannian metric.

        Parameters
        ----------
        Metric : RiemannianMetric object or instance or ScalarProductMetric instance
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

        if inspect.isclass(Metric):
            self.metric = Metric(self, **metric_kwargs)
        else:
            if self.metric._space is not self:
                raise ValueError(
                    "Cannot equip space with metric instantiated with another space."
                )

            self.metric = Metric

        return self

    def equip_with_group_action(self, group_action):
        """Equip manifold with group action.

        Parameters
        ----------
        group_action : str
            Group action.
        """
        self.group_action = group_action

        return self

    def equip_with_quotient(self):
        """Equip manifold with quotient structure.

        Creates attributes `quotient` and `fiber_bundle` or `aligner` (
        `aligner` is used in quotient contexts where the notion
        of fiber bundle is not defined.).

        Returns
        -------
        quotient : Manifold or None
            Quotient space equipped with a quotient metric.
        """
        if not _QuotientStructureRegistry.has_quotient(self):
            raise ValueError("No quotient structure defined for this manifold.")

        FiberBundle_, QuotientMetric_ = (
            _QuotientStructureRegistry.get_fiber_bundle_and_quotient_metric(
                self,
            )
        )
        fiber_bundle = FiberBundle_(total_space=self)
        if hasattr(fiber_bundle, "riemannian_submersion"):
            self.fiber_bundle = fiber_bundle
        else:
            self.aligner = fiber_bundle

        if QuotientMetric_ is None:
            return

        self.quotient = self.new(equip=False)
        self.quotient.equip_with_metric(QuotientMetric_, total_space=self)

        return self.quotient

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

    @abc.abstractmethod
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

    def random_tangent_vec(self, base_point=None, n_samples=1):
        """Generate random tangent vec.

        This method is not recommended for statistical purposes,
        as the tangent vectors generated are not drawn from a
        distribution related to the Riemannian metric.

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
            and base_point is not None
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

    def projection(self, point):
        """Project a point to the manifold.

        Parameters
        ----------
        point: array-like, shape[..., *point_shape]
            Point.

        Returns
        -------
        point: array-like, shape[..., *point_shape]
            Point.
        """
        if self.intrinsic:
            return gs.copy(point)

        raise NotImplementedError("`projection` is not implemented yet")


class _QuotientStructureRegistry:
    """Registry for quotient structures."""

    STRUCTURES = {}

    @classmethod
    def _as_key(self, Obj):
        """Transform an instance of a class into a key.

        Parameters
        ----------
        Obj : type or instance or str

        Returns
        -------
        Obj : type or str
            Hashable object as used to create dict keys
            within STRUCTURES.
        """
        if not (
            inspect.isclass(Obj)
            or isinstance(Obj, types.FunctionType)
            or isinstance(Obj, (str, tuple))
        ):
            return type(Obj)

        return Obj

    @classmethod
    def has_quotient(cls, Space):
        """Check if a given type has an associated quotient structure.

        Parameters
        ----------
        Space : type or instance or str

        Returns
        -------
        has_quotient : bool
        """
        Space = cls._as_key(Space)

        for Space_, _, _ in cls.STRUCTURES.keys():
            if Space_ is Space:
                return True
        return False

    @classmethod
    def get_available_quotients(cls, Space, Metric=None, GroupAction=None):
        """Get available quotient structures.

        Parameters
        ----------
        Space : type or instance or str
        Metric : type or instance or str
        GroupAction : type or instance of str

        Returns
        -------
        available_structures : list[tuple[type or str]]
        """
        Space = cls._as_key(Space)

        structures = []
        if Metric is None and GroupAction is None:
            for Space_, Metric_, GroupAction_ in cls.STRUCTURES.keys():
                if Space_ is Space:
                    structures.append((Metric_, GroupAction_))

            return structures

        if Metric is not None and GroupAction is None:
            Metric = cls._as_key(Metric)
            for Space_, Metric_, GroupAction_ in cls.STRUCTURES.keys():
                if Space_ is Space and Metric_ is Metric:
                    structures.append((GroupAction_,))

        if Metric is None and GroupAction is not None:
            GroupAction = cls._as_key(GroupAction)
            for Space_, Metric_, GroupAction_ in cls.STRUCTURES.keys():
                if Space_ is Space and GroupAction_ is GroupAction:
                    structures.append((Metric_,))

        return structures

    @classmethod
    def get_fiber_bundle_and_quotient_metric(cls, Space, Metric=None, GroupAction=None):
        """Get fiber bundle and quotient metric.

        Checks are done along the way. Meaningful messages with
        available structures in raised errors.

        Parameters
        ----------
        Space : type or instance or str
        Metric : type or instance or str

        Returns
        -------
        FiberBundle : type
        QuotientMetric : type
        """
        if (Metric is None or GroupAction is None) and inspect.isclass(Space):
            raise ValueError("Pass instantiated space or metric and group action info.")

        if Metric is None:
            Metric = getattr(Space, "metric", None)

        if GroupAction is None:
            GroupAction = getattr(Space, "group_action", None)

        for structure, structure_name in zip(
            [Metric, GroupAction], ["metric", "group_action"]
        ):
            if structure is None:
                available_structures = cls.get_available_quotients(
                    Space, Metric=Metric, GroupAction=GroupAction
                )
                structs_str = "\n\t".join(
                    [
                        ", ".join(str(elem) for elem in struct)
                        for struct in available_structures
                    ]
                )
                raise ValueError(
                    f"Need to equip with `{structure_name}` first. "
                    f"Available structures:\n\t{structs_str}"
                )

        Space = cls._as_key(Space)
        Metric = cls._as_key(Metric)
        GroupAction = cls._as_key(GroupAction)

        key = (Space, Metric, GroupAction)
        out = cls.STRUCTURES.get(key, None)
        if out is None:
            if isinstance(GroupAction, tuple):
                return (
                    lambda *args, **kwargs: FiberBundle(*args, **kwargs, aligner=True),
                    QuotientMetric,
                )
            else:
                raise ValueError(f"No mapping for key: {key}")

        return out


def register_quotient(Space, Metric, GroupAction, FiberBundle, QuotientMetric=None):
    """Register quotient structure.

    Parameters
    ----------
    Space : type or str
    Metric : type or str
    GroupAction : type or str
    FiberBundle : type or str
    QuotientMetric : type or str
    """
    _QuotientStructureRegistry.STRUCTURES[(Space, Metric, GroupAction)] = (
        FiberBundle,
        QuotientMetric,
    )
