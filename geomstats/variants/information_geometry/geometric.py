"""Variants of the Fisher-Rao metric on geometric distributions."""

import geomstats.backend as gs
from geomstats.geometry.open_interval import (
    ArtanhSqrtDiffeo,
    OpenIntervalEuclideanMetric,
)
from geomstats.geometry.positive_reals import PositiveReals
from geomstats.geometry.pullback_metric import CoordinatePullbackDiffeoMetric
from geomstats.geometry.riemannian_metric import CoordinateRiemannianMetric


class GeometricMetric(CoordinateRiemannianMetric):
    """Fisher-Rao metric on geometric distributions given by its metric matrix.

    The Christoffel symbols and the curvature tensors are computed from the
    metric matrix with automatic differentiation, instead of the closed forms
    of :class:`geomstats.information_geometry.geometric.GeometricMetric`.
    """

    def metric_matrix(self, base_point):
        """Compute the inner-product matrix.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., 1, 1]
            Inner-product matrix.
        """
        return gs.expand_dims(1 / (base_point**2 * (1 - base_point)), axis=-1)


class PullbackGeometricMetric(CoordinatePullbackDiffeoMetric):
    """Fisher-Rao metric on geometric distributions as a pullback metric.

    Variant of :class:`geomstats.information_geometry.geometric.GeometricMetric`
    without its closed forms: the Christoffel symbols and the curvature
    tensors are computed from the pullback metric matrix with automatic
    differentiation.
    """

    def __init__(self, space):
        super().__init__(
            space=space,
            diffeo=ArtanhSqrtDiffeo(),
            image_space=PositiveReals(equip=False).equip_with_metric(
                OpenIntervalEuclideanMetric
            ),
        )
