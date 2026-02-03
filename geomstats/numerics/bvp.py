"""Boundary value problem solvers implementation."""

import scipy

import geomstats.backend as gs
from geomstats.numerics._common import result_to_backend_type


class ScipySolveBVP:
    """Wrapper for scipy.integrate.solve_bvp.

    Parameters
    ----------
    tol : float
        Tolerance for solution.
    max_nodes : int
        Maximum number of mesh nodes.
    bc_tol : float
        Tolerance for boundary conditions.
    save_result : bool
        Whether to save result.
    """

    def __init__(self, tol=1e-3, max_nodes=1000, bc_tol=None, save_result=False):
        self.tol = tol
        self.max_nodes = max_nodes
        self.bc_tol = bc_tol

        self.save_result = save_result
        self.result_ = None

    def integrate(self, fun, bc, x, y, fun_jac=None, bc_jac=None):
        """Solve a boundary value problem for a system of ODEs.

        Parameters
        ----------
        fun : callable
            Right-hand side of the system.
        bc : callable
            Boundary conditions.
        x : array-like, shape=[n_points]
            Initial mesh.
        y : array-like, shape=[n, n_points]
            Initial guess for the solution.
        fun_jac : callable
            Jacobian of fun.
        bc_jac : callable
            Jacobian of bc.

        Returns
        -------
        result : OptimizeResult
            Solution result.
        """

        def fun_(t, state):
            return fun(t, gs.from_numpy(state))

        def bc_(state_0, state_1):
            return bc(gs.from_numpy(state_0), gs.from_numpy(state_1))

        if fun_jac is not None:

            def fun_jac_(t, state):
                return fun_jac(t, gs.from_numpy(state))

        else:
            fun_jac_ = None

        result = scipy.integrate.solve_bvp(
            fun_,
            bc_,
            x,
            y,
            tol=self.tol,
            max_nodes=self.max_nodes,
            fun_jac=fun_jac_,
            bc_jac=bc_jac,
            bc_tol=self.bc_tol,
        )

        result = result_to_backend_type(result)
        if self.save_result:
            self.result_ = result

        return result
