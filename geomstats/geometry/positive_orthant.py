"""The positive orthant."""

import math

import geomstats.backend as gs
from geomstats.geometry.base import VectorSpaceOpenSet
from geomstats.geometry.diffeo import InvolutionDiffeomorphism
from geomstats.geometry.euclidean import Euclidean, EuclideanMetric


class ScaledReciprocalDiffeo(InvolutionDiffeomorphism):
    r"""Scaled reciprocal of the second coordinate.

    .. math::

        F(x, y) = (x, x / y)

    It is an involution of the two-dimensional positive orthant.
    """

    def __call__(self, base_point):
        """Compute the scaled reciprocal of the second coordinate.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Base point.

        Returns
        -------
        image_point : array-like, shape=[..., 2]
            Image point.
        """
        return gs.stack(
            [base_point[..., 0], base_point[..., 0] / base_point[..., 1]], axis=-1
        )

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        r"""Compute the differential of the scaled reciprocal.

        .. math::

            dF_{(x, y)}(u, v) = (u, u / y - x v / y^2)

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 2]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 2]
            Base point.
        image_point : array-like, shape=[..., 2]
            Image point.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., 2]
            Image tangent vector.
        """
        if base_point is None:
            base_point = self(image_point)

        x, y = base_point[..., 0], base_point[..., 1]
        u, v = tangent_vec[..., 0], tangent_vec[..., 1]
        return gs.stack([u, u / y - x * v / y**2], axis=-1)


class PositiveOrthant(VectorSpaceOpenSet):
    """Class for the positive orthant.

    This is the set of ``dim``-tuples of positive reals.

    Parameters
    ----------
    dim : int
        Dimension of the positive orthant.
    equip : bool
        If True, equip space with default metric.
    """

    def __init__(self, dim, equip=True):
        super().__init__(
            dim=dim,
            embedding_space=Euclidean(dim=dim, equip=False),
            equip=equip,
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return PositiveOrthantEuclideanMetric

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the positive orthant.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to be checked.
        atol : float
            Tolerance.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the positive orthant.
        """
        return gs.all(point > -atol, axis=-1)

    def projection(self, point, atol=gs.atol):
        """Project a point in ambient space to the positive orthant.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in ambient space.
        atol : float
            Distance to the boundary of the projected points.

        Returns
        -------
        projected : array-like, shape=[..., dim]
            Projected point.
        """
        return gs.clip(point, atol, math.inf)

    def random_point(self, n_samples=1, bound=1.0, low=None):
        """Sample points in the positive orthant.

        The uniform distribution on [low, low + bound)^dim is used. `low`
        defaults to `gs.atol`.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        bound : float
            Length of the sampling interval in each coordinate.
        low : float
            Lower end of the sampling interval in each coordinate.

        Returns
        -------
        samples : array-like, shape=[..., dim]
            Points sampled in the positive orthant.
        """
        if low is None:
            low = gs.atol

        size = (n_samples, self.dim) if n_samples != 1 else (self.dim,)
        return gs.random.uniform(low=low, high=low + bound, size=size)


class PositiveOrthantEuclideanMetric(EuclideanMetric):
    """Class for the Euclidean metric on the positive orthant."""

    def injectivity_radius(self, base_point):
        """Compute the radius of the injectivity domain.

        It is the distance to the closest face, where a coordinate vanishes.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        return gs.amin(base_point, axis=-1)
