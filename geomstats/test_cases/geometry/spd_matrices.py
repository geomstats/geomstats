from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.geometry.matrices import MatricesMetricTestCase


class SPDMatricesTestCase(OpenSetTestCase):
    pass


class SPDEuclideanMetricTestCase(MatricesMetricTestCase):
    def test_exp_domain(self, tangent_vec, base_point, expected, atol):
        res = self.space.metric.exp_domain(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)
