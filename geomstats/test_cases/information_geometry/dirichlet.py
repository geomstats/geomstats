import pytest

import geomstats.backend as gs
from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)


class DirichletDistributionsTestCase(InformationManifoldMixinTestCase, OpenSetTestCase):
    def _check_sample_belongs_to_support(self, sample, atol):
        self.assertTrue(gs.all(sample >= 0.0))
        self.assertTrue(gs.all(gs.abs(gs.sum(sample, axis=-1) - 1.0) < atol))


class DirichletMetricTestCase(RiemannianMetricTestCase):
    def test_jacobian_christoffels(self, base_point, expected, atol):
        res = self.space.metric.jacobian_christoffels(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_sectional_curvature_is_negative(self, n_points):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        res = self.space.metric.sectional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertTrue(gs.all(res <= 0))

    @pytest.mark.random
    def test_exp_diagonal_is_totally_geodesic(self, param, tangent_param, atol):
        """Check that the diagonal x1 = ... = xn is totally geodesic."""
        base_point = param * gs.ones(self.space.shape)
        tangent_vec = tangent_param * gs.ones(self.space.shape)

        end_point = self.space.metric.exp(tangent_vec, base_point)
        self.assertAllClose(
            end_point - end_point[0], gs.zeros(self.space.shape), atol=atol
        )
