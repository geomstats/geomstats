import pytest

import geomstats.backend as gs
from geomstats.geometry.base import DiffeomorphicManifold
from geomstats.geometry.diffeo import ReversedDiffeo
from geomstats.information_geometry.gamma import (
    GammaDistributions,
    NaturalToStandardDiffeo,
)
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)


def scalar_curvature(base_point):
    kappa = base_point[..., 0]
    return (gs.polygamma(1, kappa) + kappa * gs.polygamma(2, kappa)) / (
        2 * (-1 + kappa * gs.polygamma(1, kappa)) ** 2
    )


class StandardGammaDistributions(DiffeomorphicManifold):
    """Class for the manifold of Gamma distributions in standard coordinates."""

    def __init__(self):
        super().__init__(
            diffeo=ReversedDiffeo(NaturalToStandardDiffeo()),
            image_space=GammaDistributions(equip=False),
            dim=2,
            shape=(2,),
            equip=False,
        )


class GammaDistributionsTestCase(
    InformationManifoldMixinTestCase, VectorSpaceOpenSetTestCase
):
    def _check_sample_belongs_to_support(self, sample, atol):
        self.assertTrue(gs.all(sample > 0.0))

    def test_maximum_likelihood_fit(self, data, expected, atol):
        res = self.space.maximum_likelihood_fit(data)
        self.assertAllClose(res, expected, atol=atol)


class GammaMetricTestCase(RiemannianMetricTestCase):
    def test_jacobian_christoffels(self, base_point, expected, atol):
        res = self.space.metric.jacobian_christoffels(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_scalar_curvature_against_closed_form(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        res = self.space.metric.scalar_curvature(base_point)
        expected = scalar_curvature(base_point)
        self.assertAllClose(res, expected, atol=atol)
