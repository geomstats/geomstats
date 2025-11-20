import pytest

import geomstats.backend as gs
from geomstats.test_cases.geometry.base import LevelSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)


def fisher_rao_dist(point_a, point_b, n_draws):
    return 2 * gs.sqrt(n_draws) * gs.arccos(gs.sum(gs.sqrt(point_a * point_b), axis=-1))


class MultinomialDistributionsTestCase(
    InformationManifoldMixinTestCase, LevelSetTestCase
):
    def _check_sample_belongs_to_support(self, sample, _atol):
        self.assertTrue(gs.all(sample >= 0.0))
        self.assertTrue(gs.all(gs.sum(sample, axis=-1) == self.space.n_draws))


class MultinomialMetricTestCase(RiemannianMetricTestCase):
    @pytest.mark.random
    def test_sectional_curvature_is_positive(self, n_points):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        res = self.space.metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertTrue(gs.all(res > 0))

    @pytest.mark.random
    def test_dist_against_closed_form(self, n_points, atol):
        point_a = self.data_generator.random_point(n_points)
        point_b = self.data_generator.random_point(n_points)

        res = self.space.metric.dist(point_a, point_b)
        expected = fisher_rao_dist(point_a, point_b, self.space.n_draws)
        self.assertAllClose(res, expected, atol=atol)
