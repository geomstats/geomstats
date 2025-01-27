import logging

from torchmin import minimize as torchmin_minimize

from geomstats.numerics._common import params_to_kwargs

from ._optimization import Minimizer


class TorchminMinimize(Minimizer):
    """Wrapper for torchmin.minimize.

    Check out
    https://pytorch-minimize.readthedocs.io/en/latest/api/index.html#functional-api
    for details.
    """

    def __init__(
        self,
        method="l-bfgs",
        max_iter=None,
        tol=None,
        options=None,
        callback=None,
        disp=0,
        return_all=False,
        save_result=False,
    ):
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.options = options
        self.callback = callback
        self.disp = disp
        self.return_all = return_all

        super().__init__(save_result=save_result)

    def minimize(self, fun, x0, fun_jac=None, fun_hess=None, hessp=None):
        """Minimize objective function.

        Parameters
        ----------
        fun : callable
            The objective function to be minimized.
        x0 : array-like
            Initial guess.
        fun_jac : callable
            Jacobian of fun. Ignored.
        fun_hess : callable
            Hessian of fun. Ignored.
        hessp : callable
            Ignored.
        """
        result = torchmin_minimize(
            fun, x0, **params_to_kwargs(self, func=torchmin_minimize)
        )

        if not result.success:
            logging.warning(result.message)

        if self.save_result:
            self.result_ = result

        return result
