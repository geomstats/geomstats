from geomstats.test.test_case import TestCase


class RadialKernelFunctionsTestCase(TestCase):
    def test_kernel(self, kernel, distance, bandwidth, expected, atol):
        res = kernel(distance, bandwidth)
        self.assertAllClose(res, expected, atol=atol)
