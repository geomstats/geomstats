"""The positive part of the hypersphere."""

import math

import geomstats.backend as gs
from geomstats.geometry.base import OpenSet
from geomstats.geometry.hypersphere import Hypersphere, HypersphereMetric


class PositiveHypersphere(OpenSet):
    r"""Class for the positive part of the hypersphere.

    This is the open set of the hypersphere where all the coordinates are
    positive.

    .. math::

        \left\{x \in \mathbb{R}^{n+1}: \|x\|=1
        \text { and } x_i>0 \text{ for all } i\right\}

    Parameters
    ----------
    dim : int
        Dimension of the hypersphere.
    equip : bool
        If True, equip space with default metric.
    """

    def __init__(self, dim, equip=True):
        self.dim = dim
        super().__init__(
            dim=dim,
            intrinsic=False,
            embedding_space=Hypersphere(dim),
            equip=equip,
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return PositiveHypersphereMetric

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the positive part of the hypersphere.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Point to be checked.
        atol : float
            Tolerance.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the positive part of the
            hypersphere.
        """
        is_on_sphere = self.embedding_space.belongs(point, atol=atol)
        is_positive = gs.all(point > -atol, axis=-1)
        return gs.logical_and(is_on_sphere, is_positive)

    def projection(self, point, atol=gs.atol):
        """Project a point in ambient space to the positive part of the hypersphere.

        Negative coordinates are replaced by `atol` before normalizing.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Point in ambient space.
        atol : float
            Lower bound of the coordinates before normalizing.

        Returns
        -------
        projected : array-like, shape=[..., dim + 1]
            Projected point.
        """
        return self.embedding_space.projection(gs.clip(point, atol, math.inf))

    def random_point(self, n_samples=1, bound=1.0):
        """Sample points in the positive part of the hypersphere.

        The uniform distribution is used: the coordinates of uniform points
        of the hypersphere are replaced by their absolute values.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        bound : float
            Unused.

        Returns
        -------
        samples : array-like, shape=[..., dim + 1]
            Points sampled in the positive part of the hypersphere.
        """
        return gs.abs(self.embedding_space.random_uniform(n_samples))


class PositiveHypersphereMetric(HypersphereMetric):
    """Class for the hypersphere metric on its positive part."""

    def injectivity_radius(self, base_point):
        """Compute the radius of the injectivity domain.

        It is the distance to the closest face, where a coordinate vanishes.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim + 1]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        return gs.amin(gs.arcsin(base_point), axis=-1)
