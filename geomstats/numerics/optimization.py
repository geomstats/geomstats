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

    Only `autodiff_jac` and `autodiff_hess` differ from scipy: if True, then
    automatic differentiation is used to compute jacobian and/or hessian.
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
    damped : bool
        Whether to use a damped version. Check p358 of [N2018]_.

    References
    ----------
    .. [N2018] Yurii Nesterov. Lectures on Convex Optimization. Springer, 2018.
    """

    def __init__(self, atol=gs.atol, max_iter=100, damped=False):
        self.atol = atol
        self.max_iter = max_iter
        self.damped = damped

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
            if self.damped:
                lambda_xk = gs.sqrt(gs.dot(fun_xk, y))
            else:
                lambda_xk = 0.0
            xk = xk - (1 / (1 + lambda_xk)) * y
        else:
            message = f"Maximum number of iterations {self.max_iter} reached. Results may be inaccurate"
            status = 0

        result = OptimizeResult(
            x=xk,
            success=(status == 1),
            status=status,
            method="newton",
            message=message,
            nfev=it + 1,
            njac=it,
        )

        if not result.success:
            logging.warning(result.message)

        return result
