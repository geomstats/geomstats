import math

import geomstats.algebra_utils as utils
import geomstats.backend as gs
import geomstats.tests


class TestTaylorExp(geomstats.tests.TestCase):
    def setUp(self):
        self.functions = [
            math.cos,
            lambda x: math.sin(x) / x,
            lambda x: x / math.sin(x),
            lambda x: x / math.tan(x)]
        self.coefs = [
            utils.COS_TAYLOR_COEFFS,
            utils.SINC_TAYLOR_COEFFS,
            utils.INV_SINC_TAYLOR_COEFFS,
            utils.INV_TANC_TAYLOR_COEFFS]
        self.names = [
            'cos',
            'sinc',
            'inv_sinc',
            'inv_tanc']

    def test_all(self):
        for function, coef, name in zip(
                self.functions, self.coefs, self.names):
            for exponent in range(2, 10, 2):
                x = 10 ** (-exponent)
                result = utils.taylor_exp_even_func(x, coef, function, order=5)
                expected = function(gs.sqrt(x))
                self.assertAllClose(result, expected, atol=1e-15)
