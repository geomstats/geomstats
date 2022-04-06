r"""Unit tests for the Spider."""
import geomstats.tests
from geomstats.stratified_geometry.spider import Spider, SpiderGeometry, SpiderPoint


class TestSpider(geomstats.tests.TestCase):
    space = Spider

    @geomstats.tests.np_only
    def test_belongs(self):
        rays = [10, 0, 2]
        points = [SpiderPoint(3, 13), SpiderPoint(0, 0), SpiderPoint(4, 1)]
        expected = [True, True, False]
        results = [
            (self.space(rays[k])).belongs(points[k])[0] for k in range(len(rays))
        ]
        return self.assertAllClose(expected, results)

    @geomstats.tests.np_only
    def test_random_belongs(self):
        rays = [7, 0, 12]
        results = []
        for r in rays:
            spid = self.space(r)
            p = spid.random_point(n_samples=5)
            results += [spid.belongs(p)]
        return self.assertAllClose(True, results)

    @geomstats.tests.np_only
    def test_dist(self):
        a = [SpiderPoint(10, 1), SpiderPoint(10, 2), SpiderPoint(3, 1)]
        b = [SpiderPoint(10, 31), SpiderPoint(10, 2), SpiderPoint(1, 4)]
        geom = SpiderGeometry(space=self.space(12))
        result = geom.dist(a, b)
        expected = [30, 0, 5]
        return self.assertAllClose(result, expected)

    def test_geo(self):
        a = [SpiderPoint(10, 1), SpiderPoint(10, 2), SpiderPoint(3, 1)]
        b = [SpiderPoint(10, 31), SpiderPoint(10, 2), SpiderPoint(1, 4)]
        t = [0.2]

        geom = SpiderGeometry(space=self.space(12))
        geo = geom.geodesic([a[0]], [b[0]])

        return self.assertTrue((type(geo(t)) is SpiderPoint))
