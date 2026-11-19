"""Open intervals of the real line."""

import math

import geomstats.backend as gs
from geomstats.geometry.base import VectorSpaceOpenSet
from geomstats.geometry.diffeo import Diffeo
from geomstats.geometry.euclidean import Euclidean, EuclideanMetric


class ArcsinSqrtDiffeo(Diffeo):
    r"""Scaled arcsine of the square root.

    .. math::

        F(p) = 2 \sqrt{n} \arcsin(\sqrt{p})

    It maps the interval :math:`(0, 1)` onto the interval
    :math:`(0, \pi \sqrt{n})`.

    Parameters
    ----------
    scale : float
        Scale :math:`n`.
    """

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def __call__(self, base_point):
        """Diffeomorphism at base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        image_point : array-like, shape=[..., 1]
            Image point.
        """
        return 2 * math.sqrt(self.scale) * gs.arcsin(gs.sqrt(base_point))

    def inverse(self, image_point):
        """Inverse diffeomorphism at image point.

        Parameters
        ----------
        image_point : array-like, shape=[..., 1]
            Image point.

        Returns
        -------
        base_point : array-like, shape=[..., 1]
            Base point.
        """
        return gs.sin(image_point / (2 * math.sqrt(self.scale))) ** 2

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        """Tangent diffeomorphism at base point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 1]
            Base point.
        image_point : array-like, shape=[..., 1]
            Image point.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., 1]
            Image tangent vector at image of the base point.
        """
        if base_point is None:
            base_point = self.inverse(image_point)
        return tangent_vec * gs.sqrt(self.scale / (base_point * (1 - base_point)))

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
        """Inverse tangent diffeomorphism at image point.

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., 1]
            Image tangent vector at image point.
        image_point : array-like, shape=[..., 1]
            Image point.
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point.
        """
        if base_point is None:
            base_point = self.inverse(image_point)
        return image_tangent_vec * gs.sqrt(
            base_point * (1 - base_point) / self.scale
        )


class ArtanhSqrtDiffeo(Diffeo):
    r"""Scaled hyperbolic arctangent of the square root of the complement.

    .. math::

        F(p) = 2 \operatorname{artanh}(\sqrt{1 - p})

    It maps the interval :math:`(0, 1)` onto the half-line
    :math:`(0, \infty)`.
    """

    def __call__(self, base_point):
        """Diffeomorphism at base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        image_point : array-like, shape=[..., 1]
            Image point.
        """
        return 2 * gs.arctanh(gs.sqrt(1 - base_point))

    def inverse(self, image_point):
        """Inverse diffeomorphism at image point.

        Parameters
        ----------
        image_point : array-like, shape=[..., 1]
            Image point.

        Returns
        -------
        base_point : array-like, shape=[..., 1]
            Base point.
        """
        return 1 - gs.tanh(image_point / 2) ** 2

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        """Tangent diffeomorphism at base point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 1]
            Base point.
        image_point : array-like, shape=[..., 1]
            Image point.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., 1]
            Image tangent vector at image of the base point.
        """
        if base_point is None:
            base_point = self.inverse(image_point)
        return -tangent_vec / (base_point * gs.sqrt(1 - base_point))

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
        """Inverse tangent diffeomorphism at image point.

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., 1]
            Image tangent vector at image point.
        image_point : array-like, shape=[..., 1]
            Image point.
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point.
        """
        if base_point is None:
            base_point = self.inverse(image_point)
        return -image_tangent_vec * base_point * gs.sqrt(1 - base_point)


class OpenInterval(VectorSpaceOpenSet):
    """Class for an open interval of the real line.

    The bounds may be infinite, e.g. for a half-line.

    Parameters
    ----------
    lower_bound : float
        Lower bound.
    upper_bound : float
        Upper bound.
    equip : bool
        If True, equip space with default metric.
    """

    def __init__(self, lower_bound=0.0, upper_bound=math.inf, equip=True):
        if not lower_bound < upper_bound:
            raise ValueError("The lower bound must be smaller than the upper bound.")

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        super().__init__(
            dim=1,
            embedding_space=Euclidean(dim=1, equip=False),
            equip=equip,
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return OpenIntervalMetric

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the open interval.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point to be checked.
        atol : float
            Tolerance.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the open interval.
        """
        point = point[..., 0]
        return gs.logical_and(
            self.lower_bound - atol < point, point < self.upper_bound + atol
        )

    def projection(self, point, atol=gs.atol):
        """Project a point in ambient space to the open interval.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point in ambient space.
        atol : float
            Distance to the bounds of the projected points.

        Returns
        -------
        projected : array-like, shape=[..., 1]
            Projected point.
        """
        return gs.clip(point, self.lower_bound + atol, self.upper_bound - atol)

    def random_point(self, n_samples=1, bound=1.0, low=None):
        """Sample points in the open interval.

        The uniform distribution on [low, low + bound) is used, clipped to the
        upper bound of the interval. `low` defaults to the lower bound of the
        interval if finite, and otherwise to the upper bound minus `bound`, or
        to `-bound / 2` for the real line.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        bound : float
            Length of the sampling interval.
        low : float
            Lower end of the sampling interval.

        Returns
        -------
        samples : array-like, shape=[..., 1]
            Points sampled in the open interval.
        """
        if low is None:
            if not math.isinf(self.lower_bound):
                low = self.lower_bound + gs.atol
            elif not math.isinf(self.upper_bound):
                low = self.upper_bound - bound
            else:
                low = -bound / 2

        high = min(low + bound, self.upper_bound)

        size = (n_samples, 1) if n_samples != 1 else (1,)
        return gs.random.uniform(low=low, high=high, size=size)


class OpenIntervalMetric(EuclideanMetric):
    """Class for the Euclidean metric on an open interval."""

    def injectivity_radius(self, base_point):
        """Compute the radius of the injectivity domain.

        It is the distance to the closest bound.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        point = base_point[..., 0]
        return gs.minimum(
            point - self._space.lower_bound, self._space.upper_bound - point
        )
