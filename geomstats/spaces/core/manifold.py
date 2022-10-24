"""Abstract class for manifolds.

Lead authors: Nicolas Guigui and Nina Miolane.
"""

import abc
import geomstats.backend as gs
from geomstats import errors


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
        Whether to use intrinsic (True) or extrinsic (False) coordinates
    equip_default: bool
        Whether to equip the manifold with default structure
    """

    def __init__(
        self, dim, shape, intrinsic=False, equip_default=False, **kwargs):
        super().__init__()
        errors.check_integer(dim, "dim")

        if not isinstance(shape, tuple):
            raise ValueError("Expected a tuple for the shape argument.")

        self.dim = dim
        self.shape = shape
        self.intrinsic = intrinsic
        if equip_default:
            Metric, metric_kwargs = self._default_metric()
            self.equip_metric(Metric, **metric_kwargs)

    def equip_metric(self, Metric, **kwargs):
        """
        Parameters:
            metric (cls):
        """
        if Metric is not None:
            self.metric = Metric(space=self, **kwargs)

    def equip_group_action(self, GroupAction, **kwargs):
        """
        Parameters:
            group_action (cls):
        """
        if GroupAction is not None:
            self.group_action = GroupAction(self, **kwargs)

    def equip_symmetric_structure(self, metric, group_action, **kwargs):
        """
        Equips a manifold with Riemannian symmetric structure, checking for
        compatibility between the metric and the group action.
        """
        self.metric = metric(self, **kwargs)
        self.group_action = group_action(self, **kwargs)

        def riemann_tensor(self):
            """
            Combines information from the group action with the metric
            """
            raise NotImplementedError

        self.metric.riemann_tensor = riemann_tensor

    def lie_bracket(self):
        raise NotImplementedError

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

    @abc.abstractmethod
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
        samples : array-like, shape=[..., {dim, [n, n]}]
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
        regularized_point : array-like, shape=[..., dim]
            Regularized point.
        """
        regularized_point = point
        return regularized_point

    def random_tangent_vec(self, base_point, n_samples=1):
        """Generate random tangent vec.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        base_point :  array-like, shape=[..., dim]
            Point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
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
