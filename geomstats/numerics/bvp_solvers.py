import scipy

import geomstats.backend as gs
from geomstats.numerics._common import result_to_backend_type


class ScipySolveBVP:
    """Wrapper for scipy.integrate.solve_bvp."""

    def __init__(self, tol=1e-3, max_nodes=1000, bc_tol=None, save_result=False):
        self.tol = tol
        self.max_nodes = max_nodes
        self.bc_tol = bc_tol

        self.save_result = save_result
        self.result_ = None

    def integrate(self, fun, bc, x, y, fun_jac=None, bc_jac=None):
        """Solve a boundary value problem for a system of ODEs."""

        def fun_(t, state):
            return fun(t, gs.from_numpy(state))

        def bc_(state_0, state_1):
            return bc(gs.from_numpy(state_0), gs.from_numpy(state_1))

        result = scipy.integrate.solve_bvp(
            fun_,
            bc_,
            x,
            y,
            tol=self.tol,
            max_nodes=self.max_nodes,
            fun_jac=fun_jac,
            bc_jac=bc_jac,
            bc_tol=self.bc_tol,
        )

        result = result_to_backend_type(result)
        if self.save_result:
            self.result_ = result

        return result
