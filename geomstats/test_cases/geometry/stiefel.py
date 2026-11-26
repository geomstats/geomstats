import geomstats.backend as gs
from geomstats.geometry.grassmannian import Grassmannian
from geomstats.test.assertions import assert_belongs
from geomstats.test.test_case import TestCase
from geomstats.test_cases.geometry.base import LevelSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class StiefelTestCase(LevelSetTestCase):
    def test_to_grassmannian(self, point, expected, atol=gs.atol):
        projected = self.space.to_grassmannian(point)
        self.assertAllClose(projected, expected, atol=atol)

    def test_to_grassmannian_belongs_to_grassmannian(self, point, atol=gs.atol):
        assert_belongs(
            Grassmannian(self.space.n, self.space.p, equip=False),
            self.space.to_grassmannian(point),
            atol=atol,
        )


class StiefelStaticMethodsTestCase(TestCase):
    def test_to_grassmannian(self, point, expected, atol=gs.atol):
        projected = self.Space.to_grassmannian(point)
        self.assertAllClose(projected, expected, atol=atol)


class StiefelCanonicalMetricTestCase(RiemannianMetricTestCase):
    pass
