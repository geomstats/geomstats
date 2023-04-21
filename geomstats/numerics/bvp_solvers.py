import scipy

import geomstats.backend as gs
from geomstats.numerics._common import result_to_backend_type


class ScipySolveBVP:
    def __init__(self, tol=1e-3, max_nodes=1000, save_result=False):
        self.tol = tol
        self.max_nodes = max_nodes
        self.save_result = save_result

        self.result_ = None

    def integrate(self, fun, bc, x, y):
        def bvp(t, state):
            return fun(t, gs.array(state))

        def bc_(state_0, state_1):
            return bc(gs.array(state_0), gs.array(state_1))

        result = scipy.integrate.solve_bvp(
            bvp, bc_, x, y, tol=self.tol, max_nodes=self.max_nodes
        )

        result = result_to_backend_type(result)
        if self.save_result:
            self.result_ = result

        return result
