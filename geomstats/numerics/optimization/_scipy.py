import logging

import scipy

import geomstats.backend as gs
from geomstats.exceptions import AutodiffNotImplementedError
from geomstats.numerics._common import result_to_backend_type

from ._optimization import Minimizer, RootFinder


class ScipyMinimize(Minimizer):
    """Wrapper for scipy.optimize.minimize.

    Only `autodiff_jac` and `autodiff_hess` differ from scipy: if True, then
    automatic differentiation is used to compute jacobian and/or hessian.

    Check out
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    for details.
    """

    def __init__(
        self,
        method="L-BFGS-B",
        autodiff_jac=False,
        autodiff_hess=False,
        bounds=None,
        constraints=(),
        tol=None,
        callback=None,
        options=None,
        save_result=False,
    ):
        if (autodiff_jac or autodiff_hess) and not gs.has_autodiff():
            raise AutodiffNotImplementedError(
                "Minimization with 'autodiff' requires automatic differentiation."
                "Change backend via the command "
                "export GEOMSTATS_BACKEND=pytorch in a terminal"
            )

        self.method = method
        self.autodiff_jac = autodiff_jac
        self.autodiff_hess = autodiff_hess
        self.bounds = bounds
        self.constraints = constraints
        self.tol = tol
        self.callback = callback
        self.options = options

        super().__init__(save_result=save_result)

    def _handle_jac(self, fun, fun_jac):
        if fun_jac is not None:
            fun_ = lambda x: fun(gs.from_numpy(x))
            fun_jac_ = fun_jac
            if callable(fun_jac):
                fun_jac_ = lambda x: fun_jac(gs.from_numpy(x))

            return fun_, fun_jac_

        if self.autodiff_jac:
            jac = True
            fun_ = lambda x: gs.autodiff.value_and_grad(fun)(gs.from_numpy(x))
        else:
            jac = fun_jac
            fun_ = lambda x: fun(gs.from_numpy(x))

        return fun_, jac

    def _handle_hess(self, fun, fun_hess):
        if fun_hess is not None or not self.autodiff_hess:
            fun_hess_ = fun_hess
            if callable(fun_hess):
                fun_hess_ = lambda x: fun_hess(gs.from_numpy(x))

            return fun_hess_

        return lambda x: gs.autodiff.hessian(fun)(gs.from_numpy(x))

    def minimize(self, fun, x0, fun_jac=None, fun_hess=None, hessp=None):
        """Minimize objective function.

        Parameters
        ----------
        fun : callable
            The objective function to be minimized.
        x0 : array-like
            Initial guess.
        fun_jac : callable
            Jacobian of fun.
        fun_hess : callable
            Hessian of fun.
        hessp : callable
        """
        fun_, jac = self._handle_jac(fun, fun_jac)
        hess = self._handle_hess(fun, fun_hess)

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

        if not result.success:
            logging.warning(result.message)

        if self.save_result:
            self.result_ = result

        return result


class ScipyRoot(RootFinder):
    """Wrapper for scipy.optimize.root.

    Only `autodiff_jac` differs from scipy: if True, then
    automatic differentiation is used to compute jacobian.
    """

    def __init__(
        self,
        method="hybr",
        autodiff_jac=False,
        tol=None,
        callback=None,
        options=None,
        save_result=False,
    ):
        if (autodiff_jac) and not gs.has_autodiff():
            raise AutodiffNotImplementedError(
                "Root finding with 'autodiff' requires automatic differentiation."
                "Change backend via the command "
                "export GEOMSTATS_BACKEND=pytorch in a terminal"
            )

        self.method = method
        self.autodiff_jac = autodiff_jac
        self.tol = tol
        self.callback = callback
        self.options = options
        self.save_result = save_result
        self.result_ = None

    def _handle_jac(self, fun, fun_jac):
        if fun_jac is not None:
            fun_ = lambda x: fun(gs.from_numpy(x))
            fun_jac_ = fun_jac
            if callable(fun_jac):
                fun_jac_ = lambda x: fun_jac(gs.from_numpy(x))

            return fun_, fun_jac_

        if self.autodiff_jac:
            jac = True
            fun_ = lambda x: gs.autodiff.value_and_jacobian(fun)(gs.from_numpy(x))
        else:
            jac = fun_jac
            fun_ = lambda x: fun(gs.from_numpy(x))

        return fun_, jac

    def root(self, fun, x0, fun_jac=None):
        """Find a root of a vector-valued function.

        Parameters
        ----------
        fun : callable
            Vector-valued function.
        x0 : array-like
            Initial guess.
        fun_jac : callable
            Jacobian of fun. Ignored if None.

        Returns
        -------
        res : OptimizeResult
        """
        fun_, fun_jac_ = self._handle_jac(fun, fun_jac)

        result = scipy.optimize.root(
            fun_,
            x0,
            method=self.method,
            jac=fun_jac_,
            tol=self.tol,
            callback=self.callback,
            options=self.options,
        )

        result = result_to_backend_type(result)

        if not result.success:
            logging.warning(result.message)

        if self.save_result:
            self.result_ = result

        return result
