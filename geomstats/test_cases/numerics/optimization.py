import geomstats.backend as gs
from geomstats.test.test_case import TestCase


class OptimizerTestCase(TestCase):
    """Optimizer test case."""

    def test_minimize(
        self,
        fun,
        x0,
        expected,
        fun_jac=None,
        fun_hess=None,
        hessp=None,
        atol=gs.atol,
    ):
        """Check minimization.

        Parameters
        ----------
        fun : callable
            Objective function.
        x0 : array-like
            Initial guess.
        expected : array-like
            Expected minimizer.
        fun_jac : callable or bool
            Jacobian of the objective, or True if ``fun`` returns it.
        fun_hess : callable
            Hessian of the objective.
        hessp : callable
            Hessian-vector product of the objective.
        atol : float
            Absolute tolerance.
        """
        res = self.optimizer.minimize(
            fun, x0, fun_jac=fun_jac, fun_hess=fun_hess, hessp=hessp
        )
        self.assertAllClose(res.x, expected, atol=atol)


class RootFinderTestCase(TestCase):
    """Root finder test case."""

    def test_root(self, fun, x0, expected, fun_jac=None, atol=gs.atol):
        """Check root finding.

        Parameters
        ----------
        fun : callable
            Function whose root is sought.
        x0 : array-like
            Initial guess.
        expected : array-like
            Expected root.
        fun_jac : callable
            Jacobian of the function.
        atol : float
            Absolute tolerance.
        """
        res = self.root_finder.root(fun, x0, fun_jac=fun_jac)
        self.assertAllClose(res.x, expected, atol=atol)
