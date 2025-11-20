from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class SasakiMetricTestCase(RiemannianMetricTestCase):
    def test_geodesic_discrete(self, initial_point, end_point, expected, atol):
        res = self.space.metric.geodesic_discrete(initial_point, end_point)
        self.assertAllClose(res, expected, atol=atol)
