import logging

import scipy

import geomstats.backend as gs
from geomstats.numerics._common import result_to_backend_type


class ScipyMinimize:
    def __init__(self, method="L-BFGS-B", jac=None, bounds=None,
                 tol=None, callback=None, options=None, save_result=False,):
        # TODO: fully copy scikit signature
        self.method = method
        self.options = options
        self.tol = tol
        self.jac = jac
        self.save_result = save_result

        self.callback = callback
        self.bounds = bounds

        self.result_ = None

    def optimize(self, func, x0):
        # TODO: need to improve result (e.g. vector or float if one point)

        jac = self.jac
        if self.jac == "autodiff":
            jac = True

            def func_(x):
                value, grad = gs.autodiff.value_and_grad(func, to_numpy=True)(
                    gs.from_numpy(x)
                )
                return value, grad

        else:

            def func_(x):
                value = func(gs.from_numpy(x))
                return value

        result = scipy.optimize.minimize(
            func_,
            x0,
            method=self.method,
            jac=jac,
            bounds=self.bounds,
            tol=self.tol,
            callback=self.callback,
            options=self.options,
        )

        result = result_to_backend_type(result)

        if result.status > 0:
            logging.warning(result.message)

        if self.save_result:
            self.result_ = result

        return result
