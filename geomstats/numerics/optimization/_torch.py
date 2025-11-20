import abc
import logging

from scipy.optimize import OptimizeResult
from torch.optim import LBFGS, SGD, Adam, RMSprop

import geomstats.backend as gs
from geomstats.numerics._common import params_to_kwargs

from ._optimization import Minimizer


class TorchMinimizer(Minimizer, abc.ABC):
    """Base class for torch.optim wrapper.

    Parameters
    ----------
    Topmin : torch.optim.optimizer
        Class to be instantiated.
    save_result : bool
        Whether to save result.
    """

    def __init__(self, Toptim, save_result=False):
        super().__init__(save_result=save_result)

        self._Toptim = Toptim
        self.toptim_ = None

    def _instantiate_toptim(self, x0):
        return self._Toptim([x0], **params_to_kwargs(self, func=self._Toptim))

    def _to_result(self, x0):
        return OptimizeResult(
            x=x0,
            success=1,
        )


class TorchClosuredBasedMinimizer(TorchMinimizer):
    """torch.optim wrapper based on a closure.

    Check out
    https://pytorch.org/docs/stable/optim.html#optimizer-step-closure
    for details.
    """

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

        self.toptim_ = self._instantiate_toptim(x0)

        def closure():
            self.toptim_.zero_grad()
            tobj = fun(x0)
            tobj.backward()
            return tobj

        self.toptim_.step(closure)

        result = self._to_result(x0)

        if not result.success:
            logging.warning(result.message)

        if self.save_result:
            self.result_ = result

        return result


class TorchStepwiseMinimizer(TorchMinimizer):
    """torch.optim wrapper based on a for-loop.

    Parameters
    ----------
    Topmin : torch.optim.optimizer
        Class to be instantiated.
    max_iter : int
        Maximum number of iterations.
    tol : float
        If componentwise difference between consecutive
        evaluations is less than tol, algorithm stops.
    save_result : bool
        Whether to save result.
    """

    def __init__(self, Toptim, max_iter=1000, tol=1e-7, save_result=False):
        super().__init__(Toptim, save_result=save_result)
        self.max_iter = max_iter
        self.tol = tol

        self.n_iter_ = None

    def _to_result(self, x0):
        message = "The solution converged."
        status = 1
        if self.n_iter_ == self.max_iter:
            message = (
                f"Maximum number of iterations {self.max_iter} reached."
                "Results may be inaccurate"
            )
            status = 0

        return OptimizeResult(
            x=x0,
            success=(status == 1),
            status=status,
            message=message,
            nit=self.n_iter_,
        )

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

        self.toptim_ = self._instantiate_toptim(x0)

        previous_x = gs.copy(x0)
        for iter_ in range(self.max_iter):
            self.toptim_.zero_grad()
            tobj = fun(x0)
            tobj.backward()
            self.toptim_.step()

            if ((previous_x - x0).abs() < self.tol).all():
                break

            previous_x = gs.copy(x0)

        self.n_iter_ = iter_ + 1

        result = self._to_result(x0)

        if not result.success:
            logging.warning(result.message)

        if self.save_result:
            self.result_ = result

        return result


class TorchLBFGS(TorchClosuredBasedMinimizer):
    """Wrapper for torch.optim.LBFGS.

    Check out
    https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html
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

        super().__init__(LBFGS, save_result=save_result)

    def _to_result(self, x0):
        state = self.toptim_.state_dict()["state"][0]
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

        return OptimizeResult(
            x=x0,
            success=(status == 1),
            status=status,
            message=message,
            nit=n_iter,
            nfev=nfev,
            njac=nfev,
        )


class TorchSGD(TorchStepwiseMinimizer):
    """Wrapper for torch.optim.SGD.

    Check out
    https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    for details.
    """

    def __init__(
        self,
        max_iter=1000,
        tol=1e-7,
        lr=1e-3,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        save_result=False,
    ):
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        super().__init__(
            SGD,
            max_iter=max_iter,
            tol=tol,
            save_result=save_result,
        )


class TorchRMSprop(TorchStepwiseMinimizer):
    """Wrapper for torch.optim.RMSprop.

    Check out
    https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
    for details.
    """

    def __init__(
        self,
        max_iter=1000,
        tol=1e-7,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        save_result=False,
    ):
        super().__init__(
            RMSprop,
            max_iter=max_iter,
            tol=tol,
            save_result=save_result,
        )

        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay


class TorchAdam(TorchStepwiseMinimizer):
    """Wrapper for torch.optim.Adam.

    Check out
    https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    for details.
    """

    def __init__(
        self,
        max_iter=1000,
        tol=1e-7,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        save_result=False,
    ):
        super().__init__(
            Adam,
            max_iter=max_iter,
            tol=tol,
            save_result=save_result,
        )

        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
