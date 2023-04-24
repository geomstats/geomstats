import logging

import scipy

import geomstats.backend as gs
from geomstats.numerics._common import result_to_backend_type


class ScipyMinimize:
    """Wrapper for scipy.optimize.minimize.

    Only `jac` differs from scipy: if `autodiff`, then
    `gs.autodiff.value_and_grad` is used to compute the jacobian.
    """

    def __init__(
        self,
        method="L-BFGS-B",
        jac=None,
        hess=None,
        bounds=None,
        constraints=(),
        tol=None,
        callback=None,
        options=None,
        save_result=False,
    ):
        self.method = method
        self.jac = jac
        self.hess = hess
        self.bounds = bounds
        self.constraints = constraints
        self.tol = tol
        self.callback = callback
        self.options = options

        self.save_result = save_result
        self.result_ = None

    def _handle_jac(self, fun, fun_jac):
        if fun_jac is not None:
            return fun, fun_jac

        jac = self.jac
        if self.jac == "autodiff":
            jac = True

            def fun_(x):
                value, grad = gs.autodiff.value_and_grad(fun, to_numpy=True)(
                    gs.from_numpy(x)
                )
                return value, grad

        else:

            def fun_(x):
                return fun(gs.from_numpy(x))

        return fun_, jac

    def _handle_hess(self, fun_hess):
        if fun_hess is not None:
            return fun_hess

        return self.hess

    def minimize(self, fun, x0, fun_jac=None, fun_hess=None, hessp=None):
        """Minimize objective function.

        Parameters
        ----------
        fun : callable
            The objective function to be minimized.
        x0 : array-like
            Initial guess.
        fun_jac : callable
            If not None, jac is ignored.
        fun_hess : callable
            If not None, hess is ignored.
        hessp : callable
        """
        fun_, jac = self._handle_jac(fun, fun_jac)
        hess = self._handle_hess(fun_hess)

        result = scipy.optimize.minimize(
            fun_,
            x0,
            method=self.method,
            jac=jac,
            hess=hess,
            hessp=hessp,
            bounds=self.bounds,
            tol=self.tol,
            constraints=self.constraints,
            callback=self.callback,
            options=self.options,
        )

        result = result_to_backend_type(result)

        if result.status > 0:
            logging.warning(result.message)

        if self.save_result:
            self.result_ = result

        return result
