"""Logarithmic metric on the positive reals as a pullback diffeo metric."""

import geomstats.backend as gs
from geomstats.geometry.diffeo import Diffeo
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric


class LogDiffeo(Diffeo):
    r"""Logarithm diffeomorphism from the positive reals to the reals.

    .. math::

        \log: \mathbb{R}_{>0} \rightarrow \mathbb{R}, \quad x \mapsto \log x

    with inverse the exponential function and differentials

    .. math::

        d\log_x(u) = \frac{u}{x}, \quad d\exp_z(w) = e^z w
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
        return gs.log(base_point)

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
        return gs.exp(image_point)

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

        return tangent_vec / base_point

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

        return base_point * image_tangent_vec


class PullbackPositiveRealsMetric(PullbackDiffeoMetric):
    r"""Logarithmic metric on the positive reals.

    Variant of :class:`geomstats.geometry.positive_reals.PositiveRealsMetric`
    obtained by pulling back the Euclidean metric through the logarithm:

    .. math::

        g_x(u, v) = \frac{uv}{x^2}

    Parameters
    ----------
    space : PositiveReals
        Manifold.
    """

    def __init__(self, space):
        super().__init__(
            space=space,
            diffeo=LogDiffeo(),
            image_space=Euclidean(dim=1),
        )
