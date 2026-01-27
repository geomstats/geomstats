"""Optimizers implementations."""

import logging
from abc import ABC, abstractmethod

from scipy.optimize import OptimizeResult

import geomstats.backend as gs


class Minimizer(ABC):
    """Minimizer.

    Parameters
    ----------
    save_result : bool
        Whether to save result.
    """

    def __init__(self, save_result=False):
        self.save_result = save_result

        self.result_ = None

    @abstractmethod
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
                lambda_xk = gs.sqrt(gs.einsum("...i,...i->...", fun_xk, y))
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
