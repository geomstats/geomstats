import logging

from scipy.optimize import OptimizeResult
from torch.optim import LBFGS

import geomstats.backend as gs
from geomstats.numerics._common import params_to_kwargs

from ._optimization import Minimizer


class TorchLBFGS(Minimizer):
    """Wrapper for torch.optim.LBFGS.

    Check out
    https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html#lbfgs
    for details.
    """

    def __init__(
        self,
        lr=1,
        max_iter=100,
        max_eval=None,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn=None,
        save_result=False,
    ):
        self.lr = lr
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change
        self.line_search_fn = line_search_fn

        self.history_size = 1

        self.toptim_ = None

        super().__init__(save_result=save_result)

    def _instantiate_toptim(self, x0):
        return LBFGS([x0], **params_to_kwargs(self, func=LBFGS))

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
        x0 = gs.copy(x0)
        x0.requires_grad = True

        toptim = self._instantiate_toptim(x0)

        def closure():
            toptim.zero_grad()
            tobj = fun(x0)
            tobj.backward()
            return tobj

        toptim.step(closure)
        self.toptim_ = toptim

        state = toptim.state_dict()["state"][0]
        n_iter = state.get("n_iter")
        nfev = state.get("func_evals")

        message = "The solution converged."
        status = 1
        if n_iter == self.max_iter:
            message = (
                f"Maximum number of iterations {self.max_iter} reached."
                "Results may be inaccurate"
            )
            status = 0

        if nfev == self.max_eval:
            message = (
                f"Maximum number of function evalutations {self.max_eval} reached."
                "Results may be inaccurate"
            )
            status = 0

        result = OptimizeResult(
            x=x0,
            success=(status == 1),
            status=status,
            message=message,
            nit=n_iter,
            nfev=nfev,
            njac=nfev,
        )

        if not result.success:
            logging.warning(result.message)

        if self.save_result:
            self.result_ = result

        return result
