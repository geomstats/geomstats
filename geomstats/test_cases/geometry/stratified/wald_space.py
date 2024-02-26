from geomstats.test_cases.geometry.stratified.point_set import PointTestCase


class WaldTestCase(PointTestCase):
    def test_corr(self, point, expected, atol):
        self.assertAllClose(point.corr, expected, atol=atol)
