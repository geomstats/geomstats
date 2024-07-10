"""Optimizers implementations."""

import logging
from abc import ABC, abstractmethod

import scipy
from scipy.optimize import OptimizeResult

import geomstats.backend as gs
from geomstats.exceptions import AutodiffNotImplementedError
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
        if jac == "autodiff" and gs.__name__.endswith("numpy"):
            raise AutodiffNotImplementedError(
                "Minimization with 'autodiff' requires automatic differentiation."
                "Change backend via the command "
                "export GEOMSTATS_BACKEND=pytorch in a terminal"
            )

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
                value, grad = gs.autodiff.value_and_grad(fun)(gs.from_numpy(x))
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


class RootFinder(ABC):
    """Find a root of a vector-valued function."""

    @abstractmethod
    def root(self, fun, x0, fun_jac=None):
        """Find a root of a vector-valued function.

        Parameters
        ----------
        fun : callable
            Vector-valued function.
        x0 : array-like
            Initial guess.
        fun_jac : callable
            Ignored if None.

        Returns
        -------
        res : OptimizeResult
        """


class ScipyRoot(RootFinder):
    """Wrapper for scipy.optimize.root."""

    def __init__(
        self,
        method="hybr",
        tol=None,
        callback=None,
        options=None,
        save_result=False,
    ):
        self.method = method
        self.tol = tol
        self.callback = callback
        self.options = options
        self.save_result = save_result
        self.result_ = None

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
        result = scipy.optimize.root(
            fun,
            x0,
            method=self.method,
            jac=fun_jac,
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


class NewtonMethod(RootFinder):
    """Find a root of a vector-valued function with Newton's method.

    Check https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization
    for details.

    Parameters
    ----------
    atol : float
        Tolerance to check algorithm convergence.
    max_iter : int
        Maximum iterations.
    """

    def __init__(self, atol=gs.atol, max_iter=100):
        self.atol = atol
        self.max_iter = max_iter

    def root(self, fun, x0, fun_jac):
        """Find a root of a vector-valued function.

        Parameters
        ----------
        fun : callable
            Vector-valued function.
        x0 : array-like
            Initial guess.
        fun_jac : callable
            Jacobian of fun.
        """
        xk = x0
        message = "The solution converged."
        status = 1
        for it in range(self.max_iter):
            fun_xk = fun(xk)
            if gs.linalg.norm(fun_xk) <= self.atol:
                break

            y = gs.linalg.solve(fun_jac(xk), fun_xk)
            xk = xk - y
        else:
            message = f"Maximum number of iterations {self.max_iter} reached. The mean may be inaccurate"
            status = 0

        result = OptimizeResult(
            x=xk,
            success=(status == 1),
            status=status,
            method="newton",
            message=message,
            nfev=it,
            njac=it,
        )

        if not result.success:
            logging.warning(result.message)

        return result
