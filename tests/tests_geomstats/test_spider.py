r"""Unit tests for the Spider."""
import geomstats.tests
from geomstats.stratified_geometry.spider import Spider, SpiderPoint


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
