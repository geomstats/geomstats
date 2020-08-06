import geomstats.algebra_utils as utils
import geomstats.backend as gs
import geomstats.tests


class TestTaylorExp(geomstats.tests.TestCase):
    def setUp(self):
        self.functions = [
            utils.cos_close_0,
            utils.sinc_close_0,
            utils.inv_sinc_close_0,
            utils.inv_tanc_close_0]

    def test_all(self):
        for taylor_function, coef in zip(
                self.functions, self.coefs):
            for exponent in range(2, 10, 2):
                x = 10 ** (-exponent)
                result = utils.taylor_exp_even_func(
                    x, taylor_function, order=5)
                expected = taylor_function['function'](gs.sqrt(x))
                self.assertAllClose(result, expected, atol=1e-15)
