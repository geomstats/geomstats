import math

import geomstats.algebra_utils as utils
import geomstats.tests


class TestTaylorExp(geomstats.tests.TestCase):
    def setUp(self):
        self.functions = [
            utils.cos_close_0,
            utils.sinc_close_0,
            utils.inv_sinc_close_0,
            utils.inv_tanc_close_0,
            utils.cosc_close_0]
        self.functions[4]['function'] = lambda x: (1 - math.cos(x)) / x ** 2
        # self.functions[5]['function'] = (
        #     lambda x: (1 - (x / math.tan(x))) / x ** 2)

    def test_all(self):
        for taylor_function in self.functions:
            for exponent in range(4, 12, 2):
                x = 10 ** (-exponent)
                expected = taylor_function['function'](math.sqrt(x))
                result = utils.taylor_exp_even_func(
                    x, taylor_function, order=4)
                self.assertAllClose(result, expected, atol=1e-15)
