import logging

import scipy

import geomstats.backend as gs
from geomstats.numerics._common import result_to_backend_type


class SCPMinimize:
    def __init__(self, method="L-BFGS-B", save_result=False, **options):
        self.method = method
        self.options = options
        self.save_result = save_result

        self.options.setdefault("maxiter", 25)

        self.result_ = None

    def optimize(self, func, x0, jac=None):

        if jac == "autodiff":
            jac = True

            def func_(x):
                return gs.autodiff.value_and_grad(func, to_numpy=True)(gs.array(x))

        else:

            def func_(x):
                return func(gs.array(x))

        result = scipy.optimize.minimize(
            func_,
            x0,
            method=self.method,
            jac=jac,
            options=self.options,
        )

        result = result_to_backend_type(result)

        if result.status == 1:
            logging.warning(
                "Maximum number of iterations reached. Result may be innacurate."
            )

        if self.save_result:
            self.result_ = result

        return result
