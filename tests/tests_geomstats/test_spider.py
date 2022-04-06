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
        result = []
        a = [SpiderPoint(10, 1.0), SpiderPoint(10, 2.0), SpiderPoint(3, 1.0)]
        b = [SpiderPoint(10, 31.0), SpiderPoint(10, 2.0), SpiderPoint(1, 4.0)]
        t = [0.2, 0.7, 2.0]
        geom = SpiderGeometry(space=self.space(12))
        # two input points
        geo = geom.geodesic([a[0]], [b[0]])
        # three outputs
        result += [geo(t)]
        # one initial and multiple end
        geo = geom.geodesic([a[0]], b)
        # three times three outputs
        result += [geo(t)]
        # three initial and three end
        geo = geom.geodesic(a, b)
        # three times three outputs
        result += [geo(t)]
        return self.assertTrue((type(result) is SpiderPoint))
