import pytest

import geomstats.backend as gs
from geomstats.test_cases.geometry.base import VectorSpaceTestCase
from geomstats.test_cases.geometry.flat_riemannian_metric import (
    FlatRiemannianMetricTestCase,
)
from geomstats.test_cases.geometry.mixins import GroupExpTestCaseMixins


class EuclideanTestCase(GroupExpTestCaseMixins, VectorSpaceTestCase):
    @pytest.mark.random
    def test_exp_random(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = tangent_vec + base_point

        self.test_exp(tangent_vec, base_point, expected, atol)

    @pytest.mark.random
    def test_identity_belongs(self, atol):
        self.test_belongs(self.space.identity, gs.array(True), atol)


class EuclideanMetricTestCase(FlatRiemannianMetricTestCase):
    @pytest.mark.random
    def test_cometrix_matrix_is_identity(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        res = self.space.metric.cometric_matrix(base_point)

        batch_shape = (n_points,) if n_points > 1 else ()
        expected = gs.broadcast_to(
            gs.eye(self.space.dim), batch_shape + 2 * (self.space.dim,)
        )
        self.assertAllClose(res, expected, atol=atol)
