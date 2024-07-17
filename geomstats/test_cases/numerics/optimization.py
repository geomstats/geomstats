from geomstats.test.test_case import TestCase


class OptimizerTestCase(TestCase):
    def test_minimize(
        self,
        fun,
        x0,
        expected,
        atol,
        fun_jac=None,
        fun_hess=None,
        hessp=None,
    ):
        res = self.optimizer.minimize(
            fun, x0, fun_jac=fun_jac, fun_hess=fun_hess, hessp=hessp
        )
        self.assertAllClose(res.x, expected, atol=atol)
