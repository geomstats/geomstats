import pytest

import geomstats.backend as gs
from geomstats.test_cases.geometry.base import VectorSpaceTestCase
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
