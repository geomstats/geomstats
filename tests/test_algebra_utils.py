import math

import geomstats.algebra_utils as utils
import geomstats.tests


class TestAlgebraUtils(geomstats.tests.TestCase):
    def setUp(self):
        self.functions = [
            utils.cos_close_0,
            utils.sinc_close_0,
            utils.inv_sinc_close_0,
            utils.inv_tanc_close_0,
            {'coefficients': utils.cosc_close_0['coefficients'],
             'function': lambda x: (1 - math.cos(x)) / x ** 2},
            utils.sinch_close_0,
            utils.cosh_close_0,
            {'coefficients': utils.inv_sinch_close_0['coefficients'],
             'function': lambda x: x / math.sinh(x)},
            {'coefficients': utils.inv_tanh_close_0['coefficients'],
             'function': lambda x: x / math.tanh(x)}]

    def test_all(self):
        for taylor_function in self.functions:
            for exponent in range(4, 12, 2):
                x = 10 ** (-exponent)
                expected = taylor_function['function'](math.sqrt(x))
                result = utils.taylor_exp_even_func(
                    x, taylor_function, order=4)
                self.assertAllClose(result, expected, atol=1e-15)
