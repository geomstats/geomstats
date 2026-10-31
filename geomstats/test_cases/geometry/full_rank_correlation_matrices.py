import geomstats.backend as gs
from geomstats.test.assertions import assert_belongs
from geomstats.test_cases.geometry.base import LevelSetTestCase


class FullRankCorrelationMatricesTestCase(LevelSetTestCase):
    def test_from_covariance(self, point, expected, atol=gs.atol):
        point_ = self.space.from_covariance(point)
        self.assertAllClose(point_, expected, atol=atol)

    def test_from_covariance_belongs(self, point, atol=gs.atol):
        assert_belongs(self.space, self.space.from_covariance(point), atol=atol)

    def test_diag_action(self, diagonal_vec, point, expected, atol=gs.atol):
        res = self.space.diag_action(diagonal_vec, point)
        self.assertAllClose(res, expected, atol=atol)
